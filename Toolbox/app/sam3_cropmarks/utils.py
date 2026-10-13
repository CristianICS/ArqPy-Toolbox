"""Utilities for tiled, georeferenced SAM 3 inference."""

from __future__ import annotations

import csv
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import rasterio
import rasterio.shutil
import torch
from PIL import Image
from rasterio.windows import Window

from core.raster_io import (
    LOSSLESS,
    NEAR_LOSSLESS,
    profile_for,
    rasterio_profile_kwargs,
)
from core.tiling import axis_tile_starts


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}


def parse_rgb_bands(value: str) -> tuple[int, int, int]:
    """Parse three one-based raster band numbers."""
    try:
        bands = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("RGB bands must be three comma-separated integers.") from exc
    if len(bands) != 3 or any(band < 1 for band in bands):
        raise ValueError("RGB bands must contain exactly three positive integers.")
    return bands


def iter_windows(width: int, height: int, tile_size: int, overlap: int) -> Iterable[Window]:
    """Yield full-coverage windows, including image edges."""
    for row in axis_tile_starts(height, tile_size, overlap):
        for col in axis_tile_starts(width, tile_size, overlap):
            yield Window(
                col,
                row,
                min(tile_size, width - col),
                min(tile_size, height - row),
            )


def _to_rgb_uint8(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-stretch a three-band raster tile to an RGB uint8 array."""
    rgb = np.zeros(data.shape, dtype=np.uint8)
    for index, band in enumerate(data):
        samples = band[valid & np.isfinite(band)]
        if samples.size == 0:
            continue
        low, high = np.percentile(samples, (2, 98))
        if high <= low:
            high = low + 1.0
        stretched = np.clip((band.astype(np.float32) - low) / (high - low), 0, 1)
        stretched[~np.isfinite(stretched)] = 0
        rgb[index] = np.round(stretched * 255).astype(np.uint8)
    rgb[:, ~valid] = 0
    return np.moveaxis(rgb, 0, -1)


def build_processor(confidence: float, checkpoint: Path | None = None):
    """Build the official SAM 3 image model and processor."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"device": device}
    if checkpoint is not None:
        kwargs.update(checkpoint_path=str(checkpoint), load_from_HF=False)
    model = build_sam3_image_model(**kwargs)
    return Sam3Processor(model, device=device, confidence_threshold=confidence), device


def _output_band_profile(source_profile: dict, dtype: str) -> dict:
    """A lossless single-band GTiff profile derived from a source raster's
    profile (block layout included), for the outputs that tiles are merged
    into with read-modify-write passes."""
    profile = source_profile.copy()
    profile.update(
        count=1, dtype=dtype, nodata=0,
        **rasterio_profile_kwargs(profile_for(
            dtype, tiled=False, blockxsize=None, blockysize=None,
        )),
    )
    return profile


def _encode_near_lossless(src_path: Path, dst_path: Path, product: str) -> None:
    """Copy the finished lossless `src_path` to `dst_path` as a tiled
    near-lossless GTiff. LERC is applied once here rather than in the
    read-modify-write passes, where every rewrite of a block would
    re-quantize it."""
    with rasterio.open(src_path) as src:
        dtype = src.dtypes[0]
    creation = rasterio_profile_kwargs(profile_for(dtype, product, NEAR_LOSSLESS))
    driver = creation.pop("driver")
    rasterio.shutil.copy(src_path, dst_path, driver=driver, **creation)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        # NumPy has no native bfloat16 dtype. SAM 3 can return bfloat16 scores
        # and boxes when CUDA autocast is enabled, so promote those tensors
        # before transferring them to the CPU. Keep bool masks as bool.
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
        value = value.cpu().numpy()
    return np.asarray(value)


def process_image(
    image_path: Path,
    output_folder: Path,
    processor,
    prompt: str,
    rgb_bands: tuple[int, int, int],
    tile_size: int,
    overlap: int,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
    compression: str = LOSSLESS,
) -> dict[str, int | Path]:
    """Segment one raster and write union, confidence, and instance outputs.

    `compression` (a `core.raster_io` mode) applies to the confidence
    raster only; the uint8 mask is always lossless.
    """
    mask_path = output_folder / f"{image_path.stem}_crop_marks.tif"
    confidence_path = output_folder / f"{image_path.stem}_crop_mark_confidence.tif"
    csv_path = output_folder / f"{image_path.stem}_crop_mark_instances.csv"
    outputs = (mask_path, confidence_path, csv_path)
    if not overwrite and all(path.exists() for path in outputs):
        progress(f"Skipping {image_path.name}: outputs already exist.")
        return {"image": image_path, "instances": 0, "skipped": 1}

    temp_mask = mask_path.with_suffix(".partial.tif")
    temp_confidence = confidence_path.with_suffix(".partial.tif")
    temp_csv = csv_path.with_suffix(".partial.csv")
    temp_confidence_encoded = confidence_path.with_suffix(".encoded.partial.tif")
    temporaries = (temp_mask, temp_confidence, temp_csv, temp_confidence_encoded)

    try:
        with rasterio.open(image_path) as source:
            if max(rgb_bands) > source.count:
                raise ValueError(
                    f"{image_path.name} has {source.count} band(s), but RGB band "
                    f"{max(rgb_bands)} was requested."
                )

            mask_profile = _output_band_profile(source.profile, "uint8")
            confidence_profile = _output_band_profile(source.profile, "float32")

            windows = list(iter_windows(source.width, source.height, tile_size, overlap))
            instance_count = 0
            # ``w+`` is required because overlapping tiles are merged by
            # reading the values already written to each output window.
            with rasterio.open(temp_mask, "w+", **mask_profile) as mask_dst, \
                    rasterio.open(temp_confidence, "w+", **confidence_profile) as conf_dst, \
                    temp_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                    "instance_id", "score", "tile_row", "tile_col",
                    "xmin", "ymin", "xmax", "ymax",
                ])
                writer.writeheader()
                tags = {
                    "SAM3_PROMPT": prompt,
                    "SAM3_CONFIDENCE": str(processor.confidence_threshold),
                    "RGB_BANDS": ",".join(map(str, rgb_bands)),
                }
                mask_dst.update_tags(**tags)
                conf_dst.update_tags(**tags)

                for tile_number, window in enumerate(windows, start=1):
                    progress(f"  Tile {tile_number}/{len(windows)}")
                    tile = source.read(rgb_bands, window=window, masked=True)
                    array = tile.astype(np.float32).filled(np.nan)
                    valid = ~np.any(np.ma.getmaskarray(tile), axis=0)
                    valid &= np.all(np.isfinite(array), axis=0)
                    if not valid.any():
                        continue

                    state = processor.set_image(Image.fromarray(_to_rgb_uint8(array, valid)))
                    result = processor.set_text_prompt(prompt=prompt, state=state)
                    masks = _tensor_to_numpy(result["masks"])
                    scores = _tensor_to_numpy(result["scores"]).reshape(-1)
                    boxes = _tensor_to_numpy(result["boxes"])
                    if masks.size == 0:
                        continue
                    masks = masks.reshape((-1, int(window.height), int(window.width)))

                    old_mask = mask_dst.read(1, window=window)
                    old_confidence = conf_dst.read(1, window=window)
                    union = np.any(masks, axis=0) & valid
                    confidence = np.zeros(union.shape, dtype=np.float32)
                    for instance_mask, score, box in zip(masks, scores, boxes):
                        instance_count += 1
                        confidence[instance_mask & valid] = np.maximum(
                            confidence[instance_mask & valid], float(score)
                        )
                        x0, y0, x1, y1 = box.tolist()
                        writer.writerow({
                            "instance_id": instance_count,
                            "score": f"{float(score):.6f}",
                            "tile_row": int(window.row_off),
                            "tile_col": int(window.col_off),
                            "xmin": f"{x0 + window.col_off:.2f}",
                            "ymin": f"{y0 + window.row_off:.2f}",
                            "xmax": f"{x1 + window.col_off:.2f}",
                            "ymax": f"{y1 + window.row_off:.2f}",
                        })
                    mask_dst.write(
                        np.maximum(old_mask, union.astype(np.uint8)), 1, window=window
                    )
                    conf_dst.write(
                        np.maximum(old_confidence, confidence), 1, window=window
                    )

        finished_confidence = temp_confidence
        if compression == NEAR_LOSSLESS:
            _encode_near_lossless(
                temp_confidence, temp_confidence_encoded, "confidence")
            temp_confidence.unlink()
            finished_confidence = temp_confidence_encoded

        for temporary, final in zip(
            (temp_mask, finished_confidence, temp_csv), outputs
        ):
            os.replace(temporary, final)
        return {"image": image_path, "instances": instance_count, "skipped": 0}
    except Exception:
        for path in temporaries:
            path.unlink(missing_ok=True)
        raise


def _find_images(input_path: Path) -> list[Path]:
    """Resolve a single GeoTIFF or the top-level GeoTIFFs in a folder."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Input file is not a GeoTIFF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.stem.endswith(("_crop_marks", "_crop_mark_confidence"))
        and ".partial" not in path.stem
    )
    if not images:
        raise FileNotFoundError(f"No GeoTIFF images found in {input_path}")
    return images


def process_input(
    input_path: Path,
    output_folder: Path,
    prompt: str,
    confidence: float,
    rgb_bands: tuple[int, int, int],
    tile_size: int,
    overlap: int,
    checkpoint: Path | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
    compression: str = LOSSLESS,
) -> list[dict[str, int | Path]]:
    """Load SAM 3 once, then process one GeoTIFF or a folder of GeoTIFFs."""
    images = _find_images(input_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    progress("Loading SAM 3 (the first run may download the gated checkpoint)...")
    processor, device = build_processor(confidence, checkpoint)
    progress(f"Using {device.upper()} for inference.")
    if device == "cpu":
        progress("Warning: SAM 3 CPU inference is supported but will be very slow.")

    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bfloat16 else nullcontext()
    )
    results = []
    with autocast:
        for number, image_path in enumerate(images, start=1):
            progress(f"Image {number}/{len(images)}: {image_path.name}")
            results.append(process_image(
                image_path, output_folder, processor, prompt, rgb_bands,
                tile_size, overlap, overwrite, progress, compression,
            ))
    return results


def process_folder(
    input_folder: Path,
    output_folder: Path,
    prompt: str,
    confidence: float,
    rgb_bands: tuple[int, int, int],
    tile_size: int,
    overlap: int,
    checkpoint: Path | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
    compression: str = LOSSLESS,
) -> list[dict[str, int | Path]]:
    """Backward-compatible wrapper for processing a folder of GeoTIFFs."""
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")
    return process_input(
        input_path=input_folder,
        output_folder=output_folder,
        prompt=prompt,
        confidence=confidence,
        rgb_bands=rgb_bands,
        tile_size=tile_size,
        overlap=overlap,
        checkpoint=checkpoint,
        overwrite=overwrite,
        progress=progress,
        compression=compression,
    )
