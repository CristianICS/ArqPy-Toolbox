"""`osgeo.gdal` is imported lazily, inside `correct_raster_streaming` only:
`sensors.world_view` imports `atmo_correction.utils`, which runs this
package's `__init__.py` (and therefore this module) as a side effect, and
`gui.framework` imports `sensors` unconditionally for every tool's GUI,
including tools such as `sam3_cropmarks` whose conda env has no GDAL.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import numexpr as ne
    _USE_NUMEXPR = True
except Exception:
    _USE_NUMEXPR = False

from .utils import write_log
from .formulas import FORMULAS
from atmo_correction.utils import BandParams, CtxParams, SensorParams
from core.raster_io import (
    LOSSLESS,
    RasterProfile,
    create_gtiff_like,
    dtype_range,
    numpy_dtype_to_gdal,
    profile_for,
    temp_tif,
    translate_to_cog,
)
from core.tiling import iter_blocks

@dataclass
class OutputSpec:
    """
    Output configuration for the corrected raster.

    Attributes
    ----------
    dtype : str
        Output dtype, e.g., "uint16" or "float32".
    nodata : Union[int, float]
        Output nodata value (use np.nan for float32 if desired).
    scale : Optional[float]
        If provided, multiply results by this factor (e.g., 10000 for scaled reflectance).
    clamp : Optional[Tuple[float, float]]
        (min, max) clipping range. If None, inferred from dtype if integer.
    compression : str
        `core.raster_io` compression mode ("lossless" or "near_lossless")
        of the final output. Codec and predictor are derived from the
        output dtype by `profile_for`; near-lossless only affects float
        outputs. The temporary GeoTIFF of a COG run is always lossless.
    tiled : bool
        Write tiled GeoTIFF.
    blocksize : Optional[int]
        Override block size (square) and update the creation options.
        If None, keep source blocks where possible.
    bigtiff : Optional[str]
        BIGTIFF creation option; e.g., "IF_SAFER".
    cog : Optional[bool]
        If create a Cloud Optimized Geotiff
    """
    dtype: str = "uint16"
    nodata: Union[int, float] = 0
    scale: Optional[float] = 10000.0
    clamp: Optional[Tuple[float, float]] = None
    compression: str = LOSSLESS
    tiled: bool = False
    blocksize: Optional[int] = None
    bigtiff: Optional[str] = "IF_SAFER"
    cog: Optional[bool] = True


BandFormula = Union[
    str,                                        # name in FORMULAS
    Callable[[np.ndarray, dict, dict], np.ndarray]
]


def _resolve_output_dtype(out: OutputSpec, scale_to_int: bool) -> np.dtype:
    if scale_to_int:
        return np.dtype("int16")
    return np.dtype(out.dtype)


def _resolve_clamp(out: OutputSpec, np_out_dtype: np.dtype) -> Tuple[float, float]:
    if out.clamp is not None:
        return tuple(map(float, out.clamp))
    if np.issubdtype(np_out_dtype, np.integer):
        return dtype_range(np_out_dtype)
    return -np.inf, np.inf


def _resolve_creation_profile(
    out: OutputSpec, np_out_dtype: np.dtype, band1, xsize: int, ysize: int,
) -> RasterProfile:
    """Explicit blocksize wins; otherwise preserve the source's native block
    size unless it's a full scanline; otherwise fall back to no explicit
    tiling dimensions.

    When a COG follows, this GeoTIFF is an intermediate and stays lossless
    (a second lossy pass would stack errors beyond MAX_Z_ERROR); otherwise
    it is the final output and uses `out.compression`."""
    tiled = bool(out.tiled or out.blocksize)
    blockx = blocky = None
    if out.blocksize:
        blockx = blocky = int(out.blocksize)
    else:
        bx, by = band1.GetBlockSize()
        if not (by == ysize and bx == xsize):
            blockx, blocky = bx, by
    mode = LOSSLESS if out.cog else out.compression
    return profile_for(
        np_out_dtype, "reflectance", mode,
        tiled=tiled, blockxsize=blockx, blockysize=blocky, bigtiff=out.bigtiff,
    )


def _process_block(
    src, dst, block, funcs, sensor: SensorParams, out: OutputSpec,
    band_count: int, np_out_dtype: np.dtype, clamp_min: float, clamp_max: float,
    scale_to_int: bool, str_formulas: dict,
) -> None:
    """Apply each band's formula to one window of the raster and write it
    to `dst`. Body unchanged from the original inline block loop."""
    xoff, yoff, win_xsize, win_ysize = block.xoff, block.yoff, block.xsize, block.ysize

    for bidx in range(1, band_count + 1):
        src_band = src.GetRasterBand(bidx)
        dst_band = dst.GetRasterBand(bidx)

        arr = src_band.ReadAsArray(xoff, yoff, win_xsize, win_ysize)
        if arr is None:
            raise RuntimeError(
                f"Failed reading block at x={xoff}, y={yoff}, band={bidx}"
            )

        arr = arr.astype("float32", copy=False)

        # Input nodata mask: prefer declared nodata,
        # else treat zeros as nodata
        src_nodata = src_band.GetNoDataValue()
        if src_nodata is not None and np.isfinite(src_nodata):
            nodata_val = float(src_nodata)
        else:
            nodata_val = 0
        mask = (arr == nodata_val)

        # Prepare output array filled with nodata
        out_arr = np.full(arr.shape, nodata_val, dtype=np.float32)

        # Indices of valid pixels
        valid = ~mask
        has_valid = np.any(valid)
        if has_valid:
            # Run the band-specific formula only on valid pixels
            out_valid, str_func = funcs[bidx - 1](
                arr[valid],              # only valid pixels (1D)
                sensor.bands[bidx - 1],
                sensor.ctx,
                sensor.cal_func
            )

            # Put calibrated values back into full output array
            out_arr[valid] = out_valid

            if bidx not in str_formulas.keys():
                str_formulas[bidx] = str_func

        # Optional scaling (out.scale)
        if has_valid and getattr(out, "scale", None) is not None:
            scale_val = float(out.scale)
            if _USE_NUMEXPR:
                out_arr = ne.evaluate(
                    "out_arr * scale",
                    local_dict={
                        "out_arr": out_arr,
                        "scale": scale_val
                    })
            else:
                out_arr *= scale_val

            # Scale to int if requested (valid pixels will have been
            # changed already; nodata will be restored later)
            if scale_to_int:
                out_arr = np.clip(
                    out_arr,
                    0,
                    np.iinfo(np.int16).max
                )
                # Round to nearest: a plain cast truncates, which biases
                # every value low and doubles the maximum error.
                out_arr = np.rint(out_arr).astype(np.int16, copy=False)

        # Clip
        if has_valid and (not np.isinf(clamp_min) or not np.isinf(clamp_max)):
            out_arr = np.clip(out_arr, clamp_min, clamp_max)

        # Final cast to output dtype
        out_arr = out_arr.astype(np_out_dtype, copy=False)

        # Restore nodata on masked pixels
        if np.issubdtype(np_out_dtype, np.floating) and np.isnan(out.nodata):
            out_arr[mask] = np.nan
        else:
            out_arr[mask] = out.nodata

        # Write block
        dst_band.WriteArray(out_arr, xoff, yoff)


def _finalize_cog(
    outfile_temp: Path, outfile_cog: Path, out: OutputSpec, np_out_dtype: np.dtype,
) -> None:
    """Translate the temp GeoTIFF into the final Cloud Optimized GeoTIFF.
    The caller owns (and deletes) the temp file. The COG reuses the
    overviews already built on the temp file (the COG driver's default
    `OVERVIEWS=AUTO`)."""
    translate_to_cog(outfile_temp, outfile_cog, profile_for(
        np_out_dtype, "reflectance", out.compression, bigtiff=out.bigtiff,
    ))


def correct_raster_streaming(
    infile: Union[str, Path],
    outfile: Union[str, Path],
    band_formulas: Sequence[BandFormula],
    sensor: SensorParams,
    out: Optional[OutputSpec] = None,
    build_overviews: bool = False,
    scale_to_int: bool = False
) -> None:
    """
    Apply per-band atmospheric (or radiometric) corrections efficiently
    using GDAL.

    Parameters
    ----------
    infile : str | Path
        Path to input raster (single or multi-band).
    outfile : str | Path
        Path to output corrected raster.
    band_formula : Sequence[BandFormula]
        For each band, either:
          - a string key in FORMULAS (e.g., "DOS3"), or
          - a callable f(band, params, ctx) -> float32 array.
        Length must equal the source band count.
    sensor : SensorParams
        Per-band parameter objects list and global parameters for the chosen sensor.
    out : OutputSpec, optional
        Output configuration (dtype, scaling, compression, etc.). Defaults
        to OutputSpec() when omitted.
    build_overviews : bool
        If True, builds overview levels (2,4,8,16) with average resampling.
    scale_to_int : bool, optional
        If True, multiply results by OutSpec.scale and convert to int16.
        Useful for storing reflectance-like outputs as scaled integers.

    Notes
    -----
    - Memory-safe for very large rasters: processes by native GDAL tile windows.
    - Nodata handling:
        * If the source defines `src.nodata`, those pixels are preserved as nodata.
        * Otherwise zeros are treated as nodata by default; adjust as needed by
          pre-setting a nodata in the source or customizing the mask logic below.
    - Scaling and clipping:
        * If `out.scale` is not None, results are multiplied and then clipped.
        * If `out.clamp` is None and `out.dtype` is integer, clip to dtype range.
    - GDAL's exception mode and multi-threading hints are process-global
      settings; configure them once via `core.env.configure_gdal()` at
      application startup rather than inside this function.
    """
    from osgeo import gdal

    out = out or OutputSpec()
    infile = Path(infile)
    outfile = Path(outfile)

    # Resolve callable formulas (strings -> functions)
    funcs: Tuple[Callable, ...] = tuple(
        FORMULAS[f] if isinstance(f, str) else f
        for f in band_formulas
    )

    # Store applied functions to compute the logs
    str_formulas = {}

    src = gdal.Open(str(infile), gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open input raster: {infile}")

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    band_count = src.RasterCount

    if band_count != len(funcs) or band_count != len(sensor.bands):
        src = None
        raise ValueError("The number of bands, formulas, and band params must have equal lengths")

    # Get basic geo-info
    geotransform = src.GetGeoTransform()
    projection = src.GetProjection()
    metadata = src.GetMetadata()

    # Determine output dtype (numpy + GDAL)
    np_out_dtype = _resolve_output_dtype(out, scale_to_int)
    gdal_dtype = numpy_dtype_to_gdal(np_out_dtype)

    # Determine clipping range
    clamp_min, clamp_max = _resolve_clamp(out, np_out_dtype)

    # Build any convenient derived ctx once
    ctx = replace(sensor.ctx)  # shallow copy

    band1 = src.GetRasterBand(1)
    profile = _resolve_creation_profile(out, np_out_dtype, band1, xsize, ysize)

    # Create output dataset paths
    log_path = outfile.with_suffix(".log")
    with ExitStack() as temps:
        if getattr(out, "cog", False):
            outfile_cog = outfile
            # Fresh, randomly named temp file, deleted when the `with` block
            # exits (even on error); it can never be the COG or any other
            # existing file.
            outfile = temps.enter_context(
                temp_tif(outfile.parent, f"{outfile.stem}_temp"))

        dst = create_gtiff_like(
            outfile, xsize, ysize, band_count, gdal_dtype,
            geotransform=geotransform, projection=projection, metadata=metadata,
            profile=profile,
        )
        try:
            # Set per-band nodata on output bands
            for bidx in range(1, band_count + 1):
                dst_band = dst.GetRasterBand(bidx)
                if out.nodata is not None:
                    dst_band.SetNoDataValue(float(out.nodata))

            # Determine processing block size (from output, same as input band1)
            blockx, blocky = band1.GetBlockSize()
            if blockx <= 0 or blocky <= 0:
                # Fallback to full-width scanline
                blockx, blocky = xsize, 1

            # Main windowed processing loop
            for block in iter_blocks(xsize, ysize, blockx, blocky):
                _process_block(
                    src, dst, block, funcs, sensor, out, band_count,
                    np_out_dtype, clamp_min, clamp_max, scale_to_int, str_formulas,
                )

            # Overviews
            if build_overviews:
                factors = [2, 4, 8, 16]
                # "AVERAGE" resampling similar to rasterio.enums.Resampling.average
                dst.BuildOverviews("AVERAGE", factors)
                # Optionally, you can also set overview metadata if desired:
                ovr_md = dst.GetMetadata("OVERVIEWS")
                if ovr_md is None:
                    ovr_md = {}
                ovr_md["RESAMPLING"] = "AVERAGE"
                dst.SetMetadata(ovr_md, "OVERVIEWS")

            # Flush to disk
            dst.FlushCache()
        finally:
            # Close the datasets (also on error) so the temp file can be
            # deleted on Windows.
            dst_band = None
            dst = None
            band1 = None
            src = None

        # Translate into COG
        if getattr(out, "cog", False):
            _finalize_cog(outfile, outfile_cog, out, np_out_dtype)

    # Write logs
    for i, (b_idx, str_formula) in enumerate(str_formulas.items()):
        write_log(
            sensor.bands[i],
            str_formula,
            b_idx,
            log_path
        )
