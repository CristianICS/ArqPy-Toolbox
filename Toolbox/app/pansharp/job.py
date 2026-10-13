"""Orchestrates a full classic-pansharpening run: Brovey (with WV3's LMVM
NIR2 refinement) or Bayesian, then an optional clip to a vector cutline.

Ports the per-run orchestration that used to live inline in
`gui/pansharpen.py`'s `run_job()` out of the GUI layer. Satisfies
`core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.clip import clip_raster_to_vector
from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import COMPRESSION_MODES, LOSSLESS, nodata_for

from .utils import (
    SENSORS,
    bayesian,
    brovey,
    final_cog_profile,
    output_dtype,
)

METHODS = ("Bayesian", "wBrovey")
GTIFF_SUFFIXES = (".tif", ".tiff")


@dataclass(frozen=True)
class PansharpResult:
    output_path: Path


@dataclass
class PansharpJob:
    mul_path: Path
    pan_path: Path
    out_folder: Path
    method: str
    sensor_name: str
    bayes_lambda: float = 0.995
    clip_vector: Optional[Path] = None
    compression: str = LOSSLESS

    def validate(self) -> None:
        if not self.mul_path.exists():
            raise FileNotFoundError(f"MUL image not found: {self.mul_path}")
        if not self.pan_path.exists():
            raise FileNotFoundError(f"PAN image not found: {self.pan_path}")
        for label, path in (("MUL", self.mul_path), ("PAN", self.pan_path)):
            if path.suffix.lower() not in GTIFF_SUFFIXES:
                raise ValueError(
                    f"{label} image must be a GeoTIFF (.tif/.tiff): {path}"
                )
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.method not in METHODS:
            raise ValueError(f"Unknown pansharpening method: {self.method}")
        if self.sensor_name not in SENSORS:
            raise ValueError(f"Unknown sensor: {self.sensor_name}")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")
        if self.method == "Bayesian" and not (0 <= self.bayes_lambda <= 1):
            raise ValueError(
                "The valid range of Bayes' lambda parameter is 0-1."
            )

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> PansharpResult:
        sensor = SENSORS[self.sensor_name]
        # Non-blank clip vector: pansharpen into "_temp" files first, then
        # clip them down to the final (un-suffixed) output names below.
        suffix = "_temp" if self.clip_vector is not None else ""
        # The pre-clip "_temp" files are intermediates: keep them lossless
        # so the near-lossless error is only introduced once, by the clip.
        pansharp_compression = (
            LOSSLESS if self.clip_vector is not None else self.compression
        )
        # Integer inputs give an integer product; float inputs need float32.
        dtype = output_dtype(self.mul_path, self.pan_path)

        lmnv_path = None
        if self.method == "Bayesian":
            progress("Starting Bayesian pansharpening...")
            out_path = self.out_folder / f"{self.mul_path.stem}_bayes{suffix}.tif"
            bayesian(
                self.mul_path, self.pan_path, out_path, self.bayes_lambda,
                compression=pansharp_compression, dtype=dtype,
            )
        else:
            progress("Starting weighted Brovey pansharpening...")
            out_path = self.out_folder / f"{self.mul_path.stem}_brovey{suffix}.tif"
            brovey(
                sensor, self.mul_path, self.pan_path, out_path,
                compression=pansharp_compression, dtype=dtype,
            )

            if self.sensor_name == "WV3":
                progress("Pansharpening the NIR2 band (LMVM)...")
                lmnv_path = sensor.lmnv(
                    self.mul_path, self.pan_path,
                    compression=pansharp_compression, dtype=dtype,
                    suffix=suffix,
                )

        if self.clip_vector is not None:
            progress("Clipping...")
            out_path = self._clip(out_path, suffix, dtype)
            if lmnv_path is not None:
                lmnv_path = self._clip(lmnv_path, suffix, dtype)

        # Built last, so the VRT references the final (clipped) files.
        if lmnv_path is not None:
            sensor.merge_pansharpened_bands(out_path, lmnv_path)

        return PansharpResult(output_path=out_path)

    def _clip(self, temp_path: Path, suffix: str, dtype) -> Path:
        """Clip a pre-clip "_temp" file to its final name and delete it."""
        out_clip_path = temp_path.with_stem(temp_path.stem.removesuffix(suffix))
        clip_raster_to_vector(
            temp_path, self.clip_vector, out_clip_path,
            nodata=nodata_for(dtype),
            profile=final_cog_profile(self.compression, dtype),
            as_cog=True,
        )
        temp_path.unlink()
        return out_clip_path
