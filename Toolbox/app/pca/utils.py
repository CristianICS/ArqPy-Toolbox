from pathlib import Path
import csv
import itertools
import subprocess

from osgeo import gdal

from core.env import find_otb_executable
from core.nodata import fill_nodata, mask_invalid
from core.raster_io import (
    LOSSLESS,
    gdal_creation_options,
    nodata_for,
    otb_creation_option_string,
    profile_for,
    temp_tif,
)

# OTB DimensionalityReduction writes float32.
_OTB_DTYPE = "float32"

# No-data value of the PCA outputs.
NODATA = nodata_for(_OTB_DTYPE)


def load_combis_from_csv(combi_path: str | Path) -> dict:
    """
    Load combinations from a CSV created by generate_combis() and
    reconstruct the original dictionary format.

    Parameters
    ----------
    combi_path : str
        Path to the CSV file.

    Returns
    -------
    dict
        {
            1: {'names': [...], 'pos': [...]},
            2: {'names': [...], 'pos': [...]},
            ...
        }

    Raises
    ------
    ValueError
        If the CSV is missing, malformed, or does not contain the expected
        id/bands/positions columns.
    """
    combis_dict = {}

    with open(str(combi_path), "r", newline="") as f:
        try:
            reader = csv.DictReader(f)

            for row in reader:
                idx = int(row["id"])

                names = row["bands"].split("-")
                pos = [int(p) for p in row["positions"].split("-")]

                combis_dict[idx] = {
                    "names": names,
                    "pos": pos
                }

        except (csv.Error, KeyError, ValueError) as e:
            raise ValueError(
                f"Malformed PCA combinations CSV: {combi_path}"
            ) from e

    return combis_dict


def save_combis_to_csv(combis_dict: dict, combi_path: str | Path) -> None:
    """
    Export combinations to a CSV readable by load_combis_from_csv():
    - id:        Unique combi identifier
    - bands:     Acronyms/names of bands in the combi separated by '-'
    - positions: Positions (indexes) of those bands separated by '-'
    """
    with open(str(combi_path), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'bands', 'positions'])  # header
        for idx, item in combis_dict.items():
            bands_str = '-'.join(item['names'])
            pos_str = '-'.join(str(p) for p in item['pos'])
            writer.writerow([idx, bands_str, pos_str])


def generate_combis(band_names: list, band_keys: list) -> dict:
    """
    Build every combination of 3..len-1 image bands, plus the full set.
    The ids follow this order, so they are stable for a given band list
    (the sensors' curated PCA CSVs reuse them).

    :band_names: List with band names/prefixes (e.g., ["B2","B3","B4"...]).
    :band_keys:  List with band positions aligned to band_names.

    Returns
    -------
    dict:
        {
            1: {'names': [...], 'pos': [...]},
            2: {'names': [...], 'pos': [...]},
            ...
        }
    """
    if len(band_names) != len(band_keys):
        raise ValueError(
            "band_names and band_keys must have the same length.")

    # Map each band name to its position
    name_to_pos = dict(zip(band_names, band_keys))

    # Build all combinations (size 3..len-1) + the full set
    band_combis = []
    # Calculate all combinations
    for n in range(3, len(band_names)):
        band_combis.extend(itertools.combinations(band_names, n))

    # The last combination integrates all the image bands
    band_combis.append(band_names)

    # Build final dict: {id: {'names': [...], 'pos': [...]} }
    combis_dict = {}
    for idx, combo in enumerate(band_combis, start=1):
        combis_dict[idx] = {
            'names': list(combo),
            'pos': [name_to_pos[name] for name in combo]
        }

    return combis_dict


def run_pca(
    img_path,
    out_path,
    bands,
    normalize=True,
    outmatrix=None,
    ram=None,
    nbcomp=None,  # optional: override number of components if you don't want len(bands)
):
    """
    Run PCA (OTB DimensionalityReduction) on a subset of image bands.

    A pixel with no-data in any of the bands is left out: OTB computes
    the statistics over the whole image, so those pixels are filled with
    the band means first (which keeps the means and the principal axes
    of the valid pixels) and are `NODATA` in the output.

    Parameters
    ----------
    img_path : str | PathLike
        Input multiband image.
    out_path : str | PathLike
        Output PCA image.
    bands : Sequence[int]
        1-based band indices to use for PCA (e.g., [1, 3, 5]).
    normalize : bool, default True
        Center & reduce (z-score) before PCA. (OTB's -normalize)
    outmatrix : str | None, default None
        If given, saves the transformation (eigenvectors) matrix to this file.
    ram : int | None
        Available RAM in MB for OTB streaming.
    nbcomp : int | None
        Number of principal components to keep; defaults to len(bands).
    """

    if not bands or len(bands) == 0:
        raise ValueError("`bands` must be a non-empty list of 1-based band indices.")

    otb_exe = find_otb_executable()
    out_path = Path(out_path)
    # The OTB output is a float32 intermediate (re-encoded to COG by the
    # caller), so it is written lossless whatever the run's compression.
    lossless = profile_for(_OTB_DTYPE, mode=LOSSLESS)
    otb_co = otb_creation_option_string(lossless)

    # Select the bands for the current PCA into fresh, randomly named temp
    # files (deleted when the `with` block exits, even on error).
    with temp_tif(out_path.parent, f"{out_path.stem}_sel") as temp_sel,             temp_tif(out_path.parent, f"{out_path.stem}_valid") as valid_path:
        # GDAL rather than OTB: it keeps the no-data metadata, and reads
        # the toolbox's LERC files (OTB cannot decode them).
        selection = gdal.Translate(
            str(temp_sel), str(img_path), format="GTiff",
            bandList=[int(b) for b in bands], outputType=gdal.GDT_Float32,
            creationOptions=gdal_creation_options(lossless),
        )
        if selection is None:
            raise RuntimeError(f"Could not select the bands of: {img_path}")
        selection = None
        fill_nodata([temp_sel], valid_path, fill_mean=True)

        # PCA with DimensionalityReduction
        fun_params = [
            otb_exe,
            "DimensionalityReduction",
            "-in", str(temp_sel),
            "-out", f"{out_path}?{otb_co}",
            "-method", "pca",
        ]

        if nbcomp is None:
            # Number of components: default to the number of selected bands
            fun_params.extend(["-nbcomp", f"{int(len(bands))}"])
        else:
            fun_params.extend(["-nbcomp", f"{int(nbcomp)}"])

        # Center and reduce (z-score) each band; OTB's default is false
        fun_params.extend(["-normalize", "true" if normalize else "false"])

        # Optional: export transformation matrix (eigenvectors)
        if outmatrix:
            fun_params.extend(["-outmatrix", str(outmatrix)])

        # RAM for streaming
        if ram is not None:
            fun_params.extend(["-ram", str(int(ram))])

        subprocess.run(fun_params, check=True)
        mask_invalid(out_path, valid_path, NODATA)

    if outmatrix:
        print(f"Transformation matrix saved to: {outmatrix}")
    print(f"PCA image saved to: {out_path}")
