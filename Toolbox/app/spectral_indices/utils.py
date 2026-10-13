"""Compute batches of spectral indices from a JSON index catalogue via
`gdal_calc`.

Ports the fake `Indices` "class" (its methods took no `self` and only ever
worked because every call site used it unbound, e.g. `Indices.compute_index(...)`)
into plain module-level functions, matching the same fix already applied to
`pca.utils.PCA`.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Iterable, List, Optional

from osgeo import gdal
from osgeo_utils.gdal_calc import main as gdal_calc

from core.clip import clip_raster_to_vector
from core.raster_io import (
    LOSSLESS,
    gdal_creation_options,
    profile_for,
    temp_tif,
    translate_to_cog,
)

INDICES_PATH = files("spectral_indices.data") / "indices.json"

# `gdal_calc` is run with `--type Float32`.
_INDEX_DTYPE = "float32"


class MissingBandsError(ValueError):
    """Raised when an index's required bands aren't among the sensor's
    bands. Expected for narrower sensors (not every sensor carries every
    band an index formula needs) — callers are expected to catch this and
    skip the index rather than treat it as a hard failure."""


@lru_cache(maxsize=1)
def _load_indices() -> dict:
    """Parse `indices.json` once per process instead of on every call."""
    with open(INDICES_PATH) as f:
        return json.load(f)


def get_index_keys():
    """Return all available index keys."""
    return _load_indices().keys()


def check_bands(bands_dict: dict, index_bands: Iterable[str]) -> Optional[List[str]]:
    """Return the index bands missing from `bands_dict`, or None if all are present."""
    missing = [nm for nm in index_bands if nm not in bands_dict]
    return missing if missing else None


def compute_index(
    index_key: str,
    img_path: Path,
    img_band_pos: List[int],
    img_band_names: List[str],
    out_dir: Path,
    clip_layer_path: Optional[Path] = None,
    overwrite: bool = False,
    compression: str = LOSSLESS,
) -> Optional[Path]:
    """
    Compute one spectral index for `img_path` via `gdal_calc` and write it
    as a Cloud Optimized GeoTIFF, optionally clipped to a vector cutline.

    :index_key: Index short name. Must be a key in indices.json.
    :img_path: Image path which bands will use to compute the indices.
    :img_band_pos: Position of image bands linked to img_band_names.
    :img_band_names: Image band keys (e.g. R)
    :out_dir: Directory to store the computed index.
    :clip_layer_path: An optional vector layer path to clip the index to.
    :overwrite: Recompute an index if it exists in out_dir.
    :compression: `core.raster_io` compression mode of the final COG
        (lossless, or near-lossless LERC within the "index" tolerance).
        The intermediate `gdal_calc` raster is always lossless.

    Returns the path of the computed (and, if requested, clipped) raster,
    or None if it was already computed and `overwrite` is False.

    Raises MissingBandsError if the sensor's bands don't cover what the
    index formula needs, and ValueError if the index's own formula/bands
    definition in indices.json is malformed.
    """
    # Extract the nodata value
    ds = gdal.Open(str(img_path), gdal.GA_ReadOnly)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    indices_json = _load_indices()

    # Make a name: band mapping from the two aligned lists
    if len(img_band_pos) != len(img_band_names):
        raise ValueError(
            "band_positions and band_names must be the same length")
    name_to_band = {
        nm: pos for nm, pos in zip(img_band_names, img_band_pos)}

    # Get index parameters
    index_bands = indices_json[index_key]['bands']
    formula = str(indices_json[index_key]['formula'])

    missing = check_bands(name_to_band, index_bands)
    if missing is not None:
        raise MissingBandsError(
            f"Required bands not found for index {index_key}: {missing}"
        )

    # Discover the variable order by "first appearance" in the formula
    # Consider names declared as required and match them with word
    # boundaries. Short names (like 'R') don't accidentally match inside
    # longer names (like 'RE1').
    regexp_str = r"\b(" + "|".join(map(re.escape, index_bands)) + r")\b"
    # Single regex pattern that matches any of the required band names.
    # Example if index_bands = ['N', 'R'], the pattern becomes:  \b(N|R)\b
    pattern = re.compile(regexp_str)
    # Empty list to store the variable names in the order they appear for
    #  the first time in the formula.
    seen_order = []
    # Iterate through every match of the band name in the formula string.
    # pattern.finditer(formula) returns match objects in sequence from
    # left to right.
    for m in pattern.finditer(formula):
        nm = m.group(1)  # Extract the matched band name (e.g., "N" or "R")
        # Avoid duplicates if the same name appears again
        if nm not in seen_order:
            seen_order.append(nm)  # Preserve the "first occurrence" order

    # If none were found, the index's own bands/formula definition in
    # indices.json is inconsistent — this is a data bug, not a per-sensor
    # mismatch, so it's a hard error rather than a skip.
    if not seen_order:
        raise ValueError(
            f"No required band names were found in the formula for index {index_key}."
        )

    if len(seen_order) > 26:
        raise ValueError("gdal_calc supports at most 26 variables (A..Z).")

    # Map names -> letters (A,B,C,…)
    letters = [chr(ord('A') + i) for i in range(len(seen_order))]
    name_to_letter = dict(zip(seen_order, letters))

    # Rewrite the formula safely
    # Whole-word replaces, longest names first is optional
    def sub_one(match: re.Match) -> str:
        return name_to_letter[match.group(0)]
    letter_formula = pattern.sub(sub_one, formula)

    # Construct argv
    argv = ["gdal_calc.py"]
    for nm in seen_order:
        L = name_to_letter[nm]
        band_idx = name_to_band[nm]  # 1-based
        argv += [f"-{L}", str(img_path), f"--{L}_band", str(band_idx)]

    out_path = Path(out_dir, f"{index_key}_{img_path.stem}.tif")
    if out_path.exists():
        if not overwrite:
            return None
        # `gdal.Warp` (clip) would write into an existing file, not replace it.
        out_path.unlink()

    # `gdal_calc` can't write a COG directly (the COG driver has no
    # Create()), so it writes a lossless tiled GeoTIFF first; only the
    # final COG uses `compression`.
    lossless_profile = profile_for(_INDEX_DTYPE, "index", LOSSLESS)
    final_profile = profile_for(_INDEX_DTYPE, "index", compression)

    print("formula ->", formula)
    print("name -> GDAL letter:", name_to_letter)
    print("rewritten formula:", letter_formula)

    with temp_tif(out_dir, f"{index_key}_{img_path.stem}_temp") as calc_path:
        argv += [
            "--calc", letter_formula,
            "--type", "Float32",
            "--NoDataValue", f"{nodata}",
            "--outfile", str(calc_path),
            # `temp_tif` pre-creates the (empty) file.
            "--overwrite",
        ]
        for co in gdal_creation_options(lossless_profile):
            argv += ["--co", co]

        # `gdal_calc` prints its own errors and returns 1 instead of raising.
        if gdal_calc(argv) != 0:
            raise RuntimeError(f"gdal_calc failed for index {index_key}")

        if clip_layer_path:
            clip_raster_to_vector(
                calc_path, clip_layer_path, out_path,
                nodata=nodata,
                profile=final_profile,
                as_cog=True,
            )
        else:
            translate_to_cog(calc_path, out_path, final_profile)

    return out_path
