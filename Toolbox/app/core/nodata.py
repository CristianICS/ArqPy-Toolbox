"""No-data handling around OTB applications.

OTB ignores no-data metadata and would compute with the no-data value
(e.g. -9999) as if it were data. The tools that call OTB therefore fill
the no-data of its inputs first (`fill_nodata`) and write the no-data
back into its result afterwards (`mask_invalid`).

Only the GDAL-based tools use this module, so `osgeo.gdal` is imported at
module level.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal

from .raster_io import gdal_creation_options, profile_for
from .tiling import BlockWindow, iter_blocks

# Side of the windows processed one at a time.
_BLOCK_SIZE = 2048


def _read(ds, block: BlockWindow) -> np.ndarray:
    """The window as a (bands, rows, cols) array."""
    return ds.ReadAsArray(
        block.xoff, block.yoff, block.xsize, block.ysize,
    ).reshape(-1, block.ysize, block.xsize)


def _read_has_data(ds, block: BlockWindow) -> np.ndarray:
    """Where every band of `ds` has data."""
    return np.all([
        ds.GetRasterBand(i).GetMaskBand().ReadAsArray(
            block.xoff, block.yoff, block.xsize, block.ysize) > 0
        for i in range(1, ds.RasterCount + 1)
    ], axis=0)


def fill_nodata(paths: Sequence[Union[str, Path]],
                valid_path: Union[str, Path],
                fill_mean: bool = False) -> None:
    """Replace the no-data of the rasters at `paths` (all on the same
    grid) with a fill value, in place, and write which pixels are valid to
    `valid_path` (1 = valid).

    A pixel is valid when every band of every raster has data; every other
    pixel is filled in all of them. The fill is 0, or each band's mean over
    the valid pixels with `fill_mean`: for a method that uses whole-image
    statistics, the mean biases them far less than 0 does.
    """
    datasets = [gdal.Open(str(path), gdal.GA_Update) for path in paths]
    xsize, ysize = datasets[0].RasterXSize, datasets[0].RasterYSize
    valid_ds = gdal.GetDriverByName("GTiff").Create(
        str(valid_path), xsize, ysize, 1, gdal.GDT_Byte,
        options=gdal_creation_options(profile_for("uint8")),
    )
    try:
        blocks = list(iter_blocks(xsize, ysize, _BLOCK_SIZE, _BLOCK_SIZE))

        def read_valid(block):
            return np.all(
                [_read_has_data(ds, block) for ds in datasets], axis=0)

        # Fill value per band, for each raster.
        fills = [np.zeros(ds.RasterCount) for ds in datasets]
        if fill_mean:
            count = 0
            sums = [np.zeros(ds.RasterCount) for ds in datasets]
            for block in blocks:
                valid = read_valid(block)
                count += int(valid.sum())
                for ds, band_sums in zip(datasets, sums):
                    band_sums += _read(ds, block)[:, valid].sum(
                        axis=1, dtype=np.float64)
            if count:
                fills = [band_sums / count for band_sums in sums]

        for block in blocks:
            valid = read_valid(block)
            valid_ds.GetRasterBand(1).WriteArray(
                valid.astype(np.uint8), block.xoff, block.yoff)
            if valid.all():
                continue
            for ds, fill in zip(datasets, fills):
                data = _read(ds, block)
                data[:, ~valid] = fill[:, None]
                ds.WriteArray(data, block.xoff, block.yoff)
    finally:
        # Close the datasets (also on error) so the temp files can be
        # deleted on Windows.
        datasets = valid_ds = None


def _read_valid(valid_ds, block: BlockWindow, margin: int) -> np.ndarray:
    """The valid mask of `block`, without the pixels that have a
    not-valid pixel within `margin` pixels. The outside of the image
    counts as valid, so the image border is kept."""
    left = min(margin, block.xoff)
    top = min(margin, block.yoff)
    right = min(margin, valid_ds.RasterXSize - block.xoff - block.xsize)
    bottom = min(margin, valid_ds.RasterYSize - block.yoff - block.ysize)
    valid = valid_ds.ReadAsArray(
        block.xoff - left, block.yoff - top,
        block.xsize + left + right, block.ysize + top + bottom,
    ) > 0
    if margin:
        valid = np.pad(
            valid,
            ((margin - top, margin - bottom), (margin - left, margin - right)),
            constant_values=True,
        )
        for axis in (0, 1):
            valid = sliding_window_view(
                valid, 2 * margin + 1, axis=axis).all(axis=-1)
    return valid


def mask_invalid(result_path: Union[str, Path],
                 valid_path: Union[str, Path],
                 nodata: float, margin: int = 0) -> None:
    """Write `nodata`, in place, into the pixels of `result_path` that are
    not valid in `valid_path` (see `fill_nodata`), are within `margin`
    pixels of such a pixel, or are not finite (e.g. a 0/0 ratio)."""
    result = gdal.Open(str(result_path), gdal.GA_Update)
    valid_ds = gdal.Open(str(valid_path), gdal.GA_ReadOnly)
    try:
        for block in iter_blocks(
                result.RasterXSize, result.RasterYSize,
                _BLOCK_SIZE, _BLOCK_SIZE):
            data = _read(result, block)
            invalid = (
                ~_read_valid(valid_ds, block, margin)
                | ~np.isfinite(data).all(axis=0)
            )
            if invalid.any():
                data[:, invalid] = nodata
                result.WriteArray(data, block.xoff, block.yoff)
    finally:
        # Close the datasets (also on error) so the temp files can be
        # deleted on Windows.
        result = valid_ds = None
