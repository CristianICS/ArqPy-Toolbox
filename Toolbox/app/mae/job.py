"""Orchestrates a full MAE saliency run over every GeoTIFF in a folder.

Ports the per-image tile/infer/merge/clip/stats loop that used to live
inline in `gui/mae.py`'s `run_job()` out of the GUI layer. Satisfies
`core.processor.Processor`.

Clipping uses `mae.utils.clip_mae` (fiona/rasterio-based), not
`core.clip` (GDAL-based): the `mae` conda env has no GDAL Python bindings.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import COMPRESSION_MODES, LOSSLESS

from .stats import compute_raster_stats, update_csv
from .utils import (
    DEFAULT_TILE_OVERLAP,
    clip_mae,
    compute_global_band_stats,
    compute_mae_batch,
    define_model,
    tilling,
    validate_source_bands,
)


@dataclass(frozen=True)
class MaeResult:
    output_paths: List[Path]
    stats_path: Path


@dataclass
class MaeJob:
    imgs_folder: Path
    out_folder: Path
    bands: Tuple[int, int, int]
    checkpoint: Optional[Path] = None
    clip_vector: Optional[Path] = None
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    normalization: Literal["per_tile", "global"] = "per_tile"
    compression: str = LOSSLESS

    def validate(self) -> None:
        if not self.imgs_folder.is_dir():
            raise FileNotFoundError(f"Input folder not found: {self.imgs_folder}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise FileNotFoundError(f"MAE checkpoint not found: {self.checkpoint}")
        if not 0 <= self.tile_overlap < 512:
            raise ValueError("Tile overlap must be between 0 and 511 (tile size is 512x512).")
        if self.normalization not in ("per_tile", "global"):
            raise ValueError(f"Unknown normalization mode: {self.normalization}")
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> MaeResult:
        model, device = define_model(checkpoint_path=self.checkpoint)
        stats_path = self.out_folder / "mae_stats.csv"

        output_paths: List[Path] = []
        for img_path in self.imgs_folder.glob("*.tif"):
            out_path = self.out_folder / (img_path.stem + "_mae.tif")
            if out_path.exists() or img_path.stem.endswith("_mae"):
                continue
            progress(f"Computing img {img_path.stem}")

            validate_source_bands(img_path, self.bands)
            band_stats = (
                compute_global_band_stats(img_path, self.bands)
                if self.normalization == "global" else None
            )
            tiles_meta, tiles_folder = tilling(
                img_path, overlap=self.tile_overlap, compression=True,
            )

            # With a clip, the merged mosaic stays lossless and only the
            # clipped output uses `compression`, so it is encoded once.
            compute_mae_batch(
                tiles_meta, out_path, model, device,
                bands=self.bands, band_stats=band_stats,
                compression=(
                    LOSSLESS if self.clip_vector is not None else self.compression
                ),
            )

            if self.clip_vector is not None:
                progress("Perform clip operation...")
                clip_mae(out_path, self.clip_vector, compression=self.compression)

            shutil.rmtree(tiles_folder)

            stats, _ = compute_raster_stats(out_path)
            update_csv(stats_path, stats)
            output_paths.append(out_path)

        return MaeResult(output_paths=output_paths, stats_path=stats_path)
