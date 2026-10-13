"""Orchestrates a full SAM 3 crop-mark segmentation run over one GeoTIFF or
a folder of GeoTIFFs.

Ports the validation that used to live in `gui/sam3_cropmarks.py`'s
separate `_validate()` step out of the GUI layer, so it raises through the
same `validate()`/`run()` contract every other tool now uses. Satisfies
`core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import COMPRESSION_MODES, LOSSLESS

from .utils import SUPPORTED_EXTENSIONS, process_input


@dataclass(frozen=True)
class Sam3CropmarksResult:
    processed: int
    skipped: int
    detected_instances: int
    output_folder: Path


@dataclass
class Sam3CropmarksJob:
    input_path: Path
    output_folder: Path
    prompt: str
    confidence: float
    rgb_bands: Tuple[int, int, int]
    tile_size: int
    overlap: int
    checkpoint: Optional[Path] = None
    overwrite: bool = False
    compression: str = LOSSLESS

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        if self.input_path.is_file():
            if self.input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"The selected input file must be a .tif or .tiff image: {self.input_path}"
                )
        elif not self.input_path.is_dir():
            raise ValueError("The selected input must be a GeoTIFF file or a folder.")
        if not self.output_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.output_folder}")
        if not self.prompt:
            raise ValueError("The text prompt cannot be empty.")
        if not 0 < self.confidence <= 1:
            raise ValueError("Confidence must be greater than 0 and at most 1.")
        if self.tile_size < 256:
            raise ValueError("Tile size must be at least 256 pixels.")
        if self.overlap < 0 or self.overlap >= self.tile_size:
            raise ValueError("Overlap must be non-negative and smaller than tile size.")
        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint not found: {self.checkpoint}")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> Sam3CropmarksResult:
        results = process_input(
            input_path=self.input_path,
            output_folder=self.output_folder,
            prompt=self.prompt,
            confidence=self.confidence,
            rgb_bands=self.rgb_bands,
            tile_size=self.tile_size,
            overlap=self.overlap,
            checkpoint=self.checkpoint,
            overwrite=self.overwrite,
            progress=progress,
            compression=self.compression,
        )
        skipped = sum(int(item["skipped"]) for item in results)
        return Sam3CropmarksResult(
            processed=len(results) - skipped,
            skipped=skipped,
            detected_instances=sum(int(item["instances"]) for item in results),
            output_folder=self.output_folder,
        )
