"""Shared GDAL raster creation-options and dtype-mapping helpers.

Every processing package used to hand-roll its own "TILED=YES,
COMPRESS=..." creation-options list and its own numpy<->GDAL dtype
mapping. This module is the single source of truth for both.

`osgeo.gdal` is imported lazily, inside the functions that actually
call it (`numpy_dtype_to_gdal`, `create_gtiff_like`, ...), rather than at
module level: tools with no GDAL in their conda env (e.g. `sam3_cropmarks`,
which only uses rasterio) still need the GDAL-independent helpers here
(`RasterProfile`, `gdal_creation_options`, `rasterio_profile_kwargs`, ...)
without failing to import this module at all.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np


@dataclass(frozen=True)
class RasterProfile:
    """GDAL GTiff creation-option settings shared by the processing tools."""
    compress: str = "DEFLATE"
    predictor: Optional[int] = 2
    tiled: bool = True
    blockxsize: Optional[int] = 512
    blockysize: Optional[int] = 512
    bigtiff: Optional[str] = "IF_SAFER"
    # Absolute per-pixel error bound; only meaningful with a LERC codec.
    max_z_error: Optional[float] = None
    # COG overview codec/predictor (None = same as the full-resolution
    # band) and overview resampling; only rendered by `cog_creation_options`.
    overview_compress: Optional[str] = None
    overview_predictor: Optional[int] = None
    resampling: Optional[str] = "AVERAGE"


LOSSLESS = "lossless"
NEAR_LOSSLESS = "near_lossless"
COMPRESSION_MODES = (LOSSLESS, NEAR_LOSSLESS)

# Near-lossless codec. LERC_DEFLATE rather than LERC_ZSTD: OTB's bundled
# GDAL has no ZSTD, and several outputs (atmcorr, pansharpening) are read
# back by OTB further down the pipeline.
NEAR_LOSSLESS_CODEC = "LERC_DEFLATE"

# Absolute MAX_Z_ERROR per float product type, in the product's own units.
PRODUCT_MAX_Z_ERROR = {
    "reflectance": 5e-5,  # below half a WV3 DN step (~1e-4 reflectance)
    "index": 5e-4,
    # OTB DimensionalityReduction whitens its output: every component has
    # unit variance whatever the input's scale (measured std = 1,
    # p99 - p1 = 4-5.5 on WV3, for float and scaled-int16 reflectance), so
    # 1e-3 is ~0.1% of each component's spread.
    "pca": 1e-3,
    "confidence": 1e-3,
}

# Relative MAX_Z_ERROR for products whose value range depends on the input
# (MAE saliency, high-pass responses): the bound is this factor times the
# caller-supplied `value_range`, e.g. the data's p99 - p1.
PRODUCT_RELATIVE_MAX_Z_ERROR = {
    "saliency": 1e-3,
    "highpass": 1e-3,
}

# COG overviews of a near-lossless output are written lossless with this
# codec (COG `OVERVIEW_COMPRESS`, GDAL >= 3.6): LERC's block quantization
# shows as blocky artifacts once a viewer stretches the overviews, and the
# overviews are only ~1/3 of the full-resolution pixels, so it saves little.
OVERVIEW_LOSSLESS_CODEC = "DEFLATE"


def _near_lossless_max_z_error(
    product: Optional[str], value_range: Optional[float],
) -> Optional[float]:
    """Resolve the MAX_Z_ERROR for `product`, or None if it has no
    near-lossless tolerance (unknown product, or a relative product with
    no usable `value_range`)."""
    if product in PRODUCT_MAX_Z_ERROR:
        return PRODUCT_MAX_Z_ERROR[product]
    if product in PRODUCT_RELATIVE_MAX_Z_ERROR:
        if value_range is None or not np.isfinite(value_range) or value_range <= 0:
            return None
        return PRODUCT_RELATIVE_MAX_Z_ERROR[product] * float(value_range)
    return None


def profile_for(
    dtype,
    product: Optional[str] = None,
    mode: str = LOSSLESS,
    *,
    value_range: Optional[float] = None,
    tiled: bool = True,
    blockxsize: Optional[int] = 512,
    blockysize: Optional[int] = 512,
    bigtiff: Optional[str] = "IF_SAFER",
) -> RasterProfile:
    """Build the RasterProfile for an output of `dtype`.

    Lossless: DEFLATE with the predictor derived from the dtype
    (3 = floating-point for floats, 2 = horizontal differencing for
    integers). Near-lossless: `NEAR_LOSSLESS_CODEC` with the product's
    MAX_Z_ERROR (`PRODUCT_MAX_Z_ERROR`, or `PRODUCT_RELATIVE_MAX_Z_ERROR`
    scaled by `value_range`), for float outputs only; anything without a
    tolerance falls back to lossless. COG overviews of a near-lossless
    output are lossless (`OVERVIEW_LOSSLESS_CODEC`, floating-point
    predictor). COG overviews are always resampled with AVERAGE.
    """
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"Unknown compression mode: {mode}")

    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    layout = dict(
        tiled=tiled, blockxsize=blockxsize, blockysize=blockysize,
        bigtiff=bigtiff,
    )
    if mode == NEAR_LOSSLESS and is_float:
        max_z_error = _near_lossless_max_z_error(product, value_range)
        if max_z_error is not None:
            return RasterProfile(
                compress=NEAR_LOSSLESS_CODEC, predictor=None,
                max_z_error=max_z_error,
                overview_compress=OVERVIEW_LOSSLESS_CODEC, overview_predictor=3,
                **layout,
            )
    return RasterProfile(
        compress="DEFLATE", predictor=3 if is_float else 2, **layout,
    )


def percentile_range(
    values: np.ndarray, low: float = 1.0, high: float = 99.0,
    max_samples: int = 1_000_000,
) -> float:
    """Return the `high` - `low` percentile spread of the finite `values`,
    the `value_range` for relative near-lossless products. Large inputs are
    strided down to about `max_samples` values first. NaN if no value is
    finite (`profile_for` then falls back to lossless)."""
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size > max_samples:
        values = values[::values.size // max_samples]
    p_low, p_high = np.percentile(values, (low, high))
    return float(p_high - p_low)


def gdal_creation_options(profile: RasterProfile = RasterProfile()) -> list[str]:
    """Render a RasterProfile as a GDAL GTiff driver `options=[...]` list."""
    options = []
    if profile.compress:
        options.append(f"COMPRESS={profile.compress}")
    if profile.tiled:
        options.append("TILED=YES")
    if profile.blockxsize:
        options.append(f"BLOCKXSIZE={profile.blockxsize}")
    if profile.blockysize:
        options.append(f"BLOCKYSIZE={profile.blockysize}")
    if profile.bigtiff:
        options.append(f"BIGTIFF={profile.bigtiff}")
    if profile.predictor is not None:
        options.append(f"PREDICTOR={profile.predictor}")
    if profile.max_z_error is not None:
        options.append(f"MAX_Z_ERROR={profile.max_z_error:g}")
    return options


def cog_creation_options(profile: RasterProfile = RasterProfile()) -> list[str]:
    """Render the subset of a RasterProfile accepted by GDAL's COG driver
    (COG has its own internal tiling/overview scheme, so TILED/BLOCKXSIZE/
    BLOCKYSIZE from `profile` are not passed through)."""
    options = []
    if profile.compress:
        options.append(f"COMPRESS={profile.compress}")
    if profile.predictor is not None:
        options.append(f"PREDICTOR={profile.predictor}")
    if profile.max_z_error is not None:
        options.append(f"MAX_Z_ERROR={profile.max_z_error:g}")
    if profile.overview_compress:
        options.append(f"OVERVIEW_COMPRESS={profile.overview_compress}")
    if profile.overview_predictor is not None:
        options.append(f"OVERVIEW_PREDICTOR={profile.overview_predictor}")
    if profile.resampling:
        options.append(f"RESAMPLING={profile.resampling}")
    if profile.bigtiff:
        options.append(f"BIGTIFF={profile.bigtiff}")
    return options


def rasterio_profile_kwargs(profile: RasterProfile = RasterProfile()) -> dict:
    """Render a RasterProfile as rasterio profile-update kwargs for a tiled
    GeoTIFF (rasterio's counterpart to `gdal_creation_options`)."""
    kwargs: dict = {"driver": "GTiff"}
    if profile.tiled:
        kwargs["tiled"] = True
    if profile.blockxsize:
        kwargs["blockxsize"] = profile.blockxsize
    if profile.blockysize:
        kwargs["blockysize"] = profile.blockysize
    if profile.compress:
        kwargs["compress"] = profile.compress
    if profile.predictor is not None:
        kwargs["predictor"] = profile.predictor
    if profile.max_z_error is not None:
        kwargs["max_z_error"] = profile.max_z_error
    if profile.bigtiff:
        kwargs["bigtiff"] = profile.bigtiff
    return kwargs


def rasterio_cog_kwargs(profile: RasterProfile = RasterProfile()) -> dict:
    """Render a RasterProfile as rasterio profile-update kwargs for a COG
    (rasterio's counterpart to `cog_creation_options`)."""
    kwargs: dict = {"driver": "COG"}
    for option in cog_creation_options(profile):
        key, value = option.split("=", 1)
        kwargs[key.lower()] = value
    return kwargs


def otb_creation_option_string(
    profile: RasterProfile = RasterProfile(),
    *, copy_src_overviews: bool = False, num_threads: bool = False,
) -> str:
    """Render a RasterProfile as an OTB `?&gdal:co:...` output-URI query
    string, for OTB CLI apps' `-out path?...` argument. `copy_src_overviews`
    and `num_threads` add the two GDAL creation options OTB pansharpening
    pipelines conventionally set that have no `RasterProfile` equivalent."""
    parts = [f"gdal:co:{opt}" for opt in gdal_creation_options(profile)]
    if copy_src_overviews:
        parts.append("gdal:co:COPY_SRC_OVERVIEWS=YES")
    if num_threads:
        parts.append("gdal:co:NUM_THREADS=ALL_CPUS")
    return "&" + "&".join(parts)


def numpy_dtype_to_gdal(dtype) -> int:
    """Map a numpy dtype to its GDAL type constant, defaulting to Float32."""
    from osgeo import gdal

    numpy_to_gdal = {
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("uint32"): gdal.GDT_UInt32,
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("float64"): gdal.GDT_Float64,
    }
    return numpy_to_gdal.get(np.dtype(dtype), gdal.GDT_Float32)


def dtype_range(dtype) -> tuple[float, float]:
    """Return the (min, max) representable range for a numpy dtype."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else np.finfo(dt)
    return float(info.min), float(info.max)


def create_gtiff_like(
    out_path: Union[str, Path],
    xsize: int,
    ysize: int,
    band_count: int,
    gdal_dtype: int,
    *,
    geotransform=None,
    projection: Optional[str] = None,
    metadata: Optional[dict] = None,
    profile: RasterProfile = RasterProfile(),
) -> gdal.Dataset:
    """Create a new GTiff dataset with the given geo-referencing/metadata
    and the creation options from `profile`. Raises RuntimeError if GDAL
    fails to create it (e.g. an invalid output path)."""
    from osgeo import gdal

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(out_path), xsize, ysize, band_count, gdal_dtype,
        options=gdal_creation_options(profile),
    )
    if dst is None:
        raise RuntimeError(f"Could not create output raster: {out_path}")

    if geotransform is not None:
        dst.SetGeoTransform(geotransform)
    if projection:
        dst.SetProjection(projection)
    if metadata:
        dst.SetMetadata(metadata)
    return dst


@contextmanager
def temp_tif(directory: Union[str, Path], prefix: str) -> Iterator[Path]:
    """Yield a fresh, uniquely named `<prefix>_<random>.tif` path in
    `directory`, and delete it on exit.

    The name comes from `tempfile.mkstemp`, which atomically creates a file
    that did not exist before, so the cleanup can never delete an input or
    any other existing file, whatever it is called. The file is created
    empty; GDAL/OTB overwrite it when writing to the path.
    """
    import os
    import tempfile

    fd, name = tempfile.mkstemp(
        prefix=f"{prefix}_", suffix=".tif", dir=str(directory))
    os.close(fd)
    path = Path(name)
    try:
        yield path
    except BaseException:
        # On failure a GDAL handle may still hold the file open (Windows
        # then refuses the delete); don't let that hide the real error.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    path.unlink(missing_ok=True)


def nodata_for(dtype) -> float:
    """Default no-data value for a raster of `dtype`: -9999, or 0 for
    unsigned types that cannot hold it."""
    if np.issubdtype(np.dtype(dtype), np.unsignedinteger):
        return 0
    return -9999


def gdal_type_name(dtype) -> str:
    """Return the GDAL type name (e.g. "UInt16") for a numpy dtype, as
    accepted by `gdal_translate -ot`."""
    from osgeo import gdal
    return gdal.GetDataTypeName(numpy_dtype_to_gdal(dtype))


def raster_dtype(path: Union[str, Path]) -> np.dtype:
    """Return the numpy dtype of band 1 of the raster at `path`."""
    from osgeo import gdal, gdal_array

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")
    try:
        return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(
            ds.GetRasterBand(1).DataType))
    finally:
        ds = None


def translate_to_cog(src_path: Union[str, Path], dst_path: Union[str, Path],
                      profile: RasterProfile = RasterProfile(),
                      *, nodata: Optional[float] = None,
                      dtype=None) -> None:
    """Translate an existing raster to a Cloud Optimized GeoTIFF via
    `gdal_translate`. With `dtype`, the output is converted to it (GDAL
    rounds to nearest and clamps to the type's range when converting float
    to integer). Raises CalledProcessError on failure."""
    import subprocess
    co_flags = []
    for opt in cog_creation_options(profile):
        co_flags.extend(["-co", opt])
    nodata_flags = ["-a_nodata", str(nodata)] if nodata is not None else []
    type_flags = ["-ot", gdal_type_name(dtype)] if dtype is not None else []
    subprocess.run([
        "gdal_translate",
        str(src_path),
        str(dst_path),
        *nodata_flags,
        *type_flags,
        "-of", "COG",
        *co_flags,
    ], check=True)
