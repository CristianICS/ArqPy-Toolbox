"""Geospatial grid alignment for multispectral and panchromatic rasters."""

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

from osgeo import gdal, gdal_array, osr

from core.raster_io import gdal_creation_options, profile_for


@dataclass(frozen=True)
class AlignmentResult:
    """Paths and dimensions produced by :func:`align_raster_grids`."""

    ms_path: Path
    pan_path: Path
    ms_width: int
    ms_height: int
    pan_width: int
    pan_height: int
    ratio: int
    bounds: tuple[float, float, float, float]


def _open_raster(path: Path):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"GDAL could not open raster: {path}")
    return dataset


def _validate_north_up(dataset, label: str):
    transform = dataset.GetGeoTransform()
    if transform[1] <= 0 or transform[5] >= 0:
        raise ValueError(f"{label} must have positive X and negative Y pixel sizes.")
    if abs(transform[2]) > 1e-12 or abs(transform[4]) > 1e-12:
        raise ValueError(f"Rotated {label} rasters are not supported.")
    return transform


def _extent(dataset, transform):
    left = transform[0]
    top = transform[3]
    right = left + dataset.RasterXSize * transform[1]
    bottom = top + dataset.RasterYSize * transform[5]
    return left, bottom, right, top


def _creation_options(dataset) -> list[str]:
    """Lossless DEFLATE options with the predictor derived from the band
    dtype that ``gdal.Warp`` keeps (3 for floats, 2 for integers)."""
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
        dataset.GetRasterBand(1).DataType)
    return gdal_creation_options(profile_for(
        dtype, blockxsize=None, blockysize=None, bigtiff=None,
    ))


def _same_crs(first, second) -> bool:
    first_wkt = first.GetProjection()
    second_wkt = second.GetProjection()
    if not first_wkt or not second_wkt:
        return first_wkt == second_wkt
    first_srs = osr.SpatialReference()
    second_srs = osr.SpatialReference()
    first_srs.ImportFromWkt(first_wkt)
    second_srs.ImportFromWkt(second_wkt)
    return bool(first_srs.IsSame(second_srs))


def align_raster_grids(
    ms_path: str | Path,
    pan_path: str | Path,
    output_folder: str | Path,
    ratio: int = 4,
) -> AlignmentResult:
    """Align MS and PAN to a shared extent with an exact integer ratio.

    The common spatial intersection is snapped inward to the existing MS grid.
    The PAN output then uses exactly ``ratio`` times the MS dimensions. This
    avoids the one-pixel rounding differences produced when the rasters are
    cropped independently.
    """

    ms_path = Path(ms_path)
    pan_path = Path(pan_path)
    output_folder = Path(output_folder)
    if ratio < 1:
        raise ValueError("The PAN/MS resolution ratio must be positive.")
    if not ms_path.is_file():
        raise FileNotFoundError(f"Multispectral raster not found: {ms_path}")
    if not pan_path.is_file():
        raise FileNotFoundError(f"Panchromatic raster not found: {pan_path}")
    output_folder.mkdir(parents=True, exist_ok=True)

    ms = _open_raster(ms_path)
    pan = _open_raster(pan_path)
    try:
        if pan.RasterCount != 1:
            raise ValueError("The PAN input must contain exactly one band.")
        if not _same_crs(ms, pan):
            raise ValueError("MS and PAN must use the same coordinate reference system.")

        ms_gt = _validate_north_up(ms, "MS")
        pan_gt = _validate_north_up(pan, "PAN")
        actual_ratio_x = ms_gt[1] / pan_gt[1]
        actual_ratio_y = abs(ms_gt[5] / pan_gt[5])
        if abs(actual_ratio_x - ratio) > 0.02 or abs(actual_ratio_y - ratio) > 0.02:
            raise ValueError(
                "Raster resolutions do not match the requested ratio: "
                f"X={actual_ratio_x:.6g}, Y={actual_ratio_y:.6g}, expected={ratio}."
            )

        ms_extent = _extent(ms, ms_gt)
        pan_extent = _extent(pan, pan_gt)
        overlap_left = max(ms_extent[0], pan_extent[0])
        overlap_bottom = max(ms_extent[1], pan_extent[1])
        overlap_right = min(ms_extent[2], pan_extent[2])
        overlap_top = min(ms_extent[3], pan_extent[3])
        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            raise ValueError("MS and PAN do not have a common spatial extent.")

        # Snap inward to whole pixels on the MS grid. Small epsilons prevent
        # floating-point noise at an already aligned boundary adding a pixel.
        epsilon = 1e-9
        col_start = ceil((overlap_left - ms_gt[0]) / ms_gt[1] - epsilon)
        col_end = floor((overlap_right - ms_gt[0]) / ms_gt[1] + epsilon)
        row_start = ceil((ms_gt[3] - overlap_top) / abs(ms_gt[5]) - epsilon)
        row_end = floor((ms_gt[3] - overlap_bottom) / abs(ms_gt[5]) + epsilon)
        ms_width = col_end - col_start
        ms_height = row_end - row_start
        if ms_width < 1 or ms_height < 1:
            raise ValueError("The shared aligned extent contains no complete MS pixels.")

        left = ms_gt[0] + col_start * ms_gt[1]
        right = ms_gt[0] + col_end * ms_gt[1]
        top = ms_gt[3] + row_start * ms_gt[5]
        bottom = ms_gt[3] + row_end * ms_gt[5]
        bounds = (left, bottom, right, top)
        pan_width = ms_width * ratio
        pan_height = ms_height * ratio

        aligned_ms = output_folder / "aligned_ms.tif"
        aligned_pan = output_folder / "aligned_pan.tif"
        common_options = {
            "format": "GTiff",
            "outputBounds": bounds,
            "dstSRS": ms.GetProjection(),
            "multithread": True,
            "warpOptions": ["NUM_THREADS=ALL_CPUS"],
        }
        ms_output = gdal.Warp(
            str(aligned_ms),
            ms,
            options=gdal.WarpOptions(
                width=ms_width,
                height=ms_height,
                resampleAlg="bilinear",
                creationOptions=_creation_options(ms),
                **common_options,
            ),
        )
        if ms_output is None:
            raise RuntimeError("GDAL failed to create the aligned MS raster.")
        ms_output = None

        pan_output = gdal.Warp(
            str(aligned_pan),
            pan,
            options=gdal.WarpOptions(
                width=pan_width,
                height=pan_height,
                resampleAlg="cubic",
                creationOptions=_creation_options(pan),
                **common_options,
            ),
        )
        if pan_output is None:
            raise RuntimeError("GDAL failed to create the aligned PAN raster.")
        pan_output = None
    finally:
        ms = None
        pan = None

    return AlignmentResult(
        ms_path=aligned_ms,
        pan_path=aligned_pan,
        ms_width=ms_width,
        ms_height=ms_height,
        pan_width=pan_width,
        pan_height=pan_height,
        ratio=ratio,
        bounds=bounds,
    )
