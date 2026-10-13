from contextlib import ExitStack, contextmanager
from pathlib import Path
import subprocess

import numpy as np
from osgeo import gdal

from core.env import find_otb_executable
from core.nodata import fill_nodata, mask_invalid
from core.raster_io import (
    LOSSLESS,
    gdal_creation_options,
    nodata_for,
    otb_creation_option_string,
    profile_for,
    raster_dtype,
    temp_tif,
    translate_to_cog,
)

# OTB writes float32 unless told otherwise. OTB outputs are always written
# as float32 temp files; the final product is converted to `output_dtype`.
_OTB_DTYPE = "float32"

LMVM_STEM = "B08_NIR2_Plmvm"

# LMVM matches local statistics over a window of 3 pixels around each
# pixel. Closer than that to a no-data pixel the window reaches into the
# filled area and the result is unreliable, so those pixels become no-data.
LMVM_EDGE_MARGIN = 3


def _otb_output_co() -> str:
    """OTB `-out` creation options, for float32 intermediates. They are
    always lossless: they get re-encoded later, and a second lossy pass
    would stack errors beyond MAX_Z_ERROR."""
    return otb_creation_option_string(
        profile_for(_OTB_DTYPE, "reflectance", LOSSLESS),
        copy_src_overviews=True, num_threads=True,
    )


def output_dtype(mul_img_path, pan_img_path) -> np.dtype:
    """Dtype of the pansharpened product.

    Integer inputs (raw DN, or atmcorr's scaled int16 reflectance) give an
    integer output: the fused values are in the inputs' units, so rounding
    them to the nearest integer loses nothing the inputs had. Any float
    input (e.g. atmcorr reflectance with `scale=0`) needs float32.
    """
    mul_dtype = raster_dtype(mul_img_path)
    pan_dtype = raster_dtype(pan_img_path)
    if (np.issubdtype(mul_dtype, np.integer)
            and np.issubdtype(pan_dtype, np.integer)):
        return np.promote_types(mul_dtype, pan_dtype)
    return np.dtype(_OTB_DTYPE)


def final_cog_profile(compression: str, dtype=_OTB_DTYPE):
    return profile_for(dtype, "reflectance", compression)


def _lossless_options(dtype) -> list:
    return gdal_creation_options(profile_for(dtype, mode=LOSSLESS))


def _align_to_pan(mul_img_path, pan_path, out_path, resample, bands=None):
    """Resample the MUL image (or only its `bands`) onto the PAN grid, as
    float32.

    GDAL is used rather than OTB's Superimpose because it honours no-data:
    a no-data MUL pixel is left out of the interpolation instead of being
    blended into its valid neighbours. Pixels with no valid MUL data get
    the float no-data value.
    """
    pan = gdal.Open(str(pan_path), gdal.GA_ReadOnly)
    left, xres, _, top, _, yres = pan.GetGeoTransform()
    width, height = pan.RasterXSize, pan.RasterYSize
    projection = pan.GetProjection()
    pan = None

    result = gdal.Warp(
        str(out_path), str(mul_img_path), format="GTiff",
        outputBounds=(left, top + yres * height, left + xres * width, top),
        width=width, height=height, dstSRS=projection or None,
        resampleAlg=resample, srcBands=bands,
        outputType=gdal.GDT_Float32, dstNodata=nodata_for(_OTB_DTYPE),
        multithread=True, creationOptions=_lossless_options(_OTB_DTYPE),
    )
    if result is None:
        raise RuntimeError(f"Could not align to the PAN grid: {mul_img_path}")
    result = None


@contextmanager
def _otb_inputs(mul_img_path, pan_img_path, tmp_dir, resample,
                fill_mean: bool = False, bands=None):
    """Yield `(pan, mul, valid)`: temporary copies of the inputs that OTB
    can use, and the mask of the pixels where both have data.

    `pan` is a lossless copy of the PAN image (OTB cannot decode the
    toolbox's LERC files); `mul` is the MUL image (or only its `bands`)
    resampled onto the PAN grid with `resample`. The no-data of both is
    replaced by a fill value (see `core.nodata.fill_nodata`). Every file is a fresh,
    randomly named temp file, deleted when the `with` block exits.
    """
    stem = Path(mul_img_path).stem
    with ExitStack() as temps:
        pan_path = temps.enter_context(temp_tif(tmp_dir, f"{stem}_pan"))
        mul_path = temps.enter_context(temp_tif(tmp_dir, f"{stem}_onpan"))
        valid_path = temps.enter_context(temp_tif(tmp_dir, f"{stem}_valid"))

        pan_copy = gdal.Translate(
            str(pan_path), str(pan_img_path), format="GTiff",
            creationOptions=_lossless_options(raster_dtype(pan_img_path)),
        )
        if pan_copy is None:
            raise RuntimeError(f"Could not copy the PAN image: {pan_img_path}")
        pan_copy = None

        _align_to_pan(mul_img_path, pan_path, mul_path, resample, bands)
        fill_nodata([pan_path, mul_path], valid_path, fill_mean)
        yield pan_path, mul_path, valid_path


def _write_product(result_path, valid_path, out_path, compression, dtype,
                   margin: int = 0):
    """Write the OTB result as the final COG of `dtype`, with no-data
    where an input had no data (`valid_path`), within `margin` pixels of
    such a pixel, or where the result is not finite (a 0/0 ratio)."""
    nodata = nodata_for(dtype)
    mask_invalid(result_path, valid_path, nodata, margin)
    translate_to_cog(
        result_path, out_path, final_cog_profile(compression, dtype),
        nodata=nodata, dtype=dtype,
    )


class Sensor:
    def __init__(self, name: str, weights: list):
        self.name = name
        self.weights = weights

    def compute_brovey_formula(self):
        """For Brovey transformation, convert weights into OTB format."""
        n = len(self.weights)

        # Denominator: sum_i w_i * im2b(i+1)
        den = "+".join(
            f"{self.weights[i]}*im2b{i+1}" for i in range(n)
        )

        # Construct the RCS operation by band
        # Each of this operations will be divided by the above den
        # Numerators: im1b1 * im2b(i+1) / (den). The weights only build
        # the synthetic PAN of the denominator (as in gdal_pansharpen):
        # weighting the numerator too scales each band by its weight.
        nums = [
            f"(im1b1*im2b{i+1}) / ({den})"
            for i in range(n)
        ]

        # Final expression: { num_1, num_2, ..., num_n }
        # NOTE: double braces for scaping, OTB requires the expression
        # to be inside braces
        expr = f"{{ {','.join(nums)} }}"
        return expr


class WV3(Sensor):

    def lmnv(self, mul_img_path: Path, pan_img_path: Path,
             compression: str = LOSSLESS, dtype=_OTB_DTYPE, suffix: str = ""):
        """Pansharpen the NIR2 band with LMVM. `suffix` is appended to the
        output stem (the job's pre-clip "_temp" marker)."""

        otb_exe = find_otb_executable()
        otb_co = _otb_output_co()

        tmp_dir = mul_img_path.parent
        # ONLY band 8 (NIR2) of the MUL image, on the PAN grid.
        with _otb_inputs(
            mul_img_path, pan_img_path, tmp_dir, "bilinear", bands=[8],
        ) as (pan_in, nir2_pan_path, valid_path), \
                temp_tif(tmp_dir, "WV3_NIR2_lmvm") as lmvm_temp_path:

            # Pansharpen ONLY the NIR2 band using LMVM
            # (local mean/variance with HPF PAN)
            # Built from the stem, so it can never equal the input path
            # (a plain `name.replace("DOS3.tif", ...)` returned the input
            # name unchanged for inputs not named *DOS3.tif).
            stem = mul_img_path.stem
            prefix = stem[:-len("DOS3")] if stem.endswith("DOS3") else f"{stem}_"
            out_path = mul_img_path.with_name(
                f"{prefix}{LMVM_STEM}{suffix}.tif")
            subprocess.run([
                otb_exe,
                "Pansharpening",
                # Panchromatic (0.31 m)
                "-inp", str(pan_in),
                # Single-band NIR2 (1.24 m)
                "-inxs", str(nir2_pan_path),
                # Use defaults window: 3x3
                "-method", "lmvm",
                "-out", f"{lmvm_temp_path}?{otb_co}"
            ], check=True)

            _write_product(
                lmvm_temp_path, valid_path, out_path, compression, dtype,
                margin=LMVM_EDGE_MARGIN,
            )

        return out_path

    def merge_pansharpened_bands(self, brovey_path, lmnv_path):
        """Create VRT with the bands from Brovey and LMNV."""
        out_name = lmnv_path.stem.removesuffix(LMVM_STEM) + "HIGHRES_stack.vrt"

        subprocess.check_call([
            "gdalbuildvrt",
            "-separate",
            str(lmnv_path.with_name(out_name)),
            str(brovey_path),
            str(lmnv_path)
        ])


WV3 = WV3("WV3", [0.005, 0.142, 0.209, 0.144, 0.234, 0.157, 0.116])
LEGION = Sensor("LEGION", [])

SENSORS = {"WV3": WV3, "LEGION": LEGION}


def brovey(
    sensor: Sensor,
    mul_img_path,
    pan_img_path,
    out_path,
    compression: str = LOSSLESS,
    dtype=_OTB_DTYPE,
):
    """Run weighted Brovey pansharpen operation. OTB computes in float32;
    the final COG is converted to `dtype` (see `output_dtype`)."""
    otb_exe = find_otb_executable()
    otb_co = _otb_output_co()

    tmp_dir = out_path.parent
    # The MUL image aligned with the PAN dimensions.
    with _otb_inputs(
        mul_img_path, pan_img_path, tmp_dir, "bilinear",
    ) as (pan_in, mul_temp_path, valid_path), \
            temp_tif(tmp_dir, f"{out_path.stem}_btemp") as out_path_temp:

        # Output the OTB expression
        expr = sensor.compute_brovey_formula()

        # Export the image with OTB, then convert to COG with GDAL
        subprocess.run([
            otb_exe,
            "BandMathX",
            "-il", str(pan_in), str(mul_temp_path),
            "-out", f"{out_path_temp}?{otb_co}",
            "-exp", expr,
        ], check=True)

        _write_product(out_path_temp, valid_path, out_path, compression, dtype)


def bayesian(mul_img_path, pan_img_path, out_path, bayes_lambda=0.995,
             compression: str = LOSSLESS, dtype=_OTB_DTYPE):
    """
    OTB CLI command for Bayesian pansharpening using the Pansharpening application.
    OTB computes in float32; the final COG is converted to `dtype` (see
    `output_dtype`).
        """
    otb_exe = find_otb_executable()
    otb_co = _otb_output_co()

    tmp_dir = out_path.parent
    # The MUL image aligned with the PAN dimensions. OTB computes the
    # Bayesian statistics over the whole image, so no-data is filled with
    # the band means.
    with _otb_inputs(
        mul_img_path, pan_img_path, tmp_dir, "near", fill_mean=True,
    ) as (pan_in, mul_temp_path, valid_path), \
            temp_tif(tmp_dir, f"{out_path.stem}_btemp") as out_temp_path:

        # Compute Bayesian pansharpening
        subprocess.run([
            otb_exe,
            "Pansharpening",
            "-inp", str(pan_in),
            "-inxs", str(mul_temp_path),
            "-out", f"{out_temp_path}?{otb_co}", "float",
            "-method", "bayes",
            "-method.bayes.lambda", str(bayes_lambda),
            "-method.bayes.s", "1"
        ], check=True)

        _write_product(out_temp_path, valid_path, out_path, compression, dtype)
