"""Orchestrates a full atmospheric-correction run for a sensor image folder.

Ports the per-image discovery/clip/correct/cleanup loop that used to live
inline in `gui/atmcorr.py`'s `run_job()` out of the GUI layer. Satisfies
`core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type

from core.clip import clip_raster_to_vector
from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import (
    COMPRESSION_MODES,
    LOSSLESS,
    RasterProfile,
    nodata_for,
    raster_dtype,
    temp_tif,
)

from .formulas import FORMULAS
from .streaming import OutputSpec, correct_raster_streaming
from .utils import SensorParams


@dataclass(frozen=True)
class AtmoCorrectionResult:
    corrected_paths: List[Path]


@dataclass
class AtmoCorrectionJob:
    image_dir: Path
    out_folder: Path
    sensor_cls: Type
    sensor_name: str
    atm_formula: str
    scale: int
    clip_vector: Optional[Path] = None
    compression: str = LOSSLESS

    def validate(self) -> None:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.atm_formula not in FORMULAS:
            raise ValueError(f"Unknown atmospheric correction formula: {self.atm_formula}")
        if self.scale < 0:
            raise ValueError("The scale value must be positive.")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> AtmoCorrectionResult:
        image = self.sensor_cls(self.image_dir)

        scale = None if self.scale == 0 else self.scale
        scale_to_integer = scale is not None and scale > 0

        progress(f"Running atmospheric correction on: {self.image_dir.stem}")

        corrected_paths: List[Path] = []
        for img_type, img_path in image.get_images().items():
            init_message = f"Correcting image {img_path.stem}"
            progress(init_message)
            progress("-" * len(init_message) * 2)

            if self.clip_vector is None:
                self._correct(image, img_type, img_path, img_path,
                              scale, scale_to_integer, corrected_paths, progress)
                continue

            # The clip intermediate is a fresh, randomly named temp file,
            # deleted when the `with` block exits (even on error), so it can
            # never be an input or any other existing file.
            with temp_tif(self.out_folder, f"{img_path.stem}_temp") as out_clip_path:
                progress("Clipping...")
                # Lossless intermediate of the raw (integer) DN image, so
                # PREDICTOR=2 is the right predictor here. Its no-data must
                # fit the DN dtype (0 for unsigned DN): the correction masks
                # the pixels equal to the declared no-data.
                clip_raster_to_vector(
                    img_path, self.clip_vector, out_clip_path,
                    nodata=nodata_for(raster_dtype(img_path)),
                    profile=RasterProfile(
                        compress="DEFLATE", predictor=2, tiled=True,
                        blockxsize=None, blockysize=None, bigtiff=None,
                    ),
                    as_cog=False,
                )
                self._correct(image, img_type, img_path, out_clip_path,
                              scale, scale_to_integer, corrected_paths, progress)
                progress("Removing temp files...")

        return AtmoCorrectionResult(corrected_paths=corrected_paths)

    def _correct(self, image, img_type: str, img_path: Path,
                 correction_input: Path, scale: Optional[int],
                 scale_to_integer: bool, corrected_paths: List[Path],
                 progress: Progress) -> None:
        """Correct `correction_input` (the raw image or its clipped temp
        copy); metadata and the output name come from `img_path`."""
        if self.atm_formula != "DOS3":
            return

        if img_type == "mul":
            image.START_BAND = "C"
        elif img_type == "pan":
            image.START_BAND = "P"

        band_params = image.extract_params_per_band(img_path)
        ctx_params = image.extract_ctx_params(img_path)

        params = SensorParams(
            self.sensor_name, band_params, ctx_params,
            self.sensor_cls.radiometric_calibration,
        )

        outname = img_path.stem + f"_{self.atm_formula}.tif"
        out_path = self.out_folder / outname

        progress("Starting DOS3 correction...")
        band_formulas = [self.atm_formula] * len(band_params)
        correct_raster_streaming(
            infile=correction_input,
            outfile=out_path,
            band_formulas=band_formulas,
            sensor=params,
            out=OutputSpec(
                dtype="float32",
                nodata=-9999,
                scale=scale,
                compression=self.compression,
                blocksize=512,
                tiled=True,
                cog=True,
            ),
            build_overviews=True,
            scale_to_int=scale_to_integer,
        )
        corrected_paths.append(out_path)
