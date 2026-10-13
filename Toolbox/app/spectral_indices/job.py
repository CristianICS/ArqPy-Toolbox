"""Orchestrates a full spectral-indices batch run: compute every index the
sensor's bands support, skip the ones they don't, and optionally clip each
to a vector cutline.

Ports the per-index loop that used to live inline in `gui/spectral_indices.py`'s
`run_job()` out of the GUI layer. Satisfies `core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import COMPRESSION_MODES, LOSSLESS

from .utils import MissingBandsError, compute_index, get_index_keys


@dataclass(frozen=True)
class SpectralIndicesResult:
    computed_paths: List[Path]


@dataclass
class SpectralIndicesJob:
    img_path: Path
    out_folder: Path
    sensor_band_names: List[str]
    sensor_band_pos: List[int]
    clip_vector: Optional[Path] = None
    overwrite: bool = False
    compression: str = LOSSLESS

    def validate(self) -> None:
        if not self.img_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.img_path}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> SpectralIndicesResult:
        computed_paths: List[Path] = []
        for index_key in get_index_keys():
            progress(f"\n{index_key}")
            progress("-" * 20)
            try:
                out_path = compute_index(
                    index_key,
                    self.img_path,
                    self.sensor_band_pos,
                    self.sensor_band_names,
                    self.out_folder,
                    clip_layer_path=self.clip_vector,
                    overwrite=self.overwrite,
                    compression=self.compression,
                )
            except MissingBandsError as e:
                progress(f"Skipping {index_key}: {e}")
                continue

            if out_path is not None:
                computed_paths.append(out_path)

        return SpectralIndicesResult(computed_paths=computed_paths)
