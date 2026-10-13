"""Orchestrates a single-band high-pass filter run: compute all five
filter outputs and optionally clip them to a vector cutline.

Ports the resolved-input orchestration that used to live inline in
`gui/highpass.py`'s `run_job()` out of the GUI layer. Satisfies
`core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import COMPRESSION_MODES, LOSSLESS

from .utils import GDALHighPassFilter


@dataclass(frozen=True)
class HighPassResult:
    output_paths: List[Path]


@dataclass
class HighPassJob:
    img_path: Path
    out_folder: Path
    band: int
    clip_vector: Optional[Path] = None
    compression: str = LOSSLESS

    def validate(self) -> None:
        if not self.img_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.img_path}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.band < 1:
            raise ValueError("The band index must be a positive integer.")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> HighPassResult:
        progress(f"Computing high-pass filters on: {self.img_path.stem}")
        hp = GDALHighPassFilter(
            self.img_path, band_index=self.band, compression=self.compression,
        )
        clip_arg = str(self.clip_vector) if self.clip_vector is not None else False
        output_paths = list(hp.run_all(self.out_folder / self.img_path.stem, clip_arg))
        return HighPassResult(output_paths=output_paths)
