from pathlib import Path

import PySimpleGUI as sg

from mae import DEFAULT_TILE_OVERLAP, parse_mae_bands
from mae.job import MaeJob

from .framework import (
    ToolWindow,
    clip_vector_row,
    compression_row,
    folder_input_row,
    resolve_clip_vector_value,
    resolve_compression_value,
)

_NORMALIZATION_LABELS = {
    "Per-tile (default)": "per_tile",
    "Global stats": "global",
}


def _parse_overlap(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError("The tile overlap must be an integer number.")


def run_job(values: dict):
    """Compute MAE saliency maps for every GeoTIFF image in a folder."""
    try:
        raw_folder = values.get("-FOLDER-", "")
        if not raw_folder:
            return ValueError("Please select a valid input folder.")
        imgs_folder = Path(raw_folder)
        if not imgs_folder.is_dir():
            return FileNotFoundError(f"Input folder not found: {imgs_folder}")

        bands = parse_mae_bands(values["-BANDS-"])
        tile_overlap = _parse_overlap(values["-OVERLAP-"])
        normalization = _NORMALIZATION_LABELS[values["-NORMALIZATION-"]]

        checkpoint_text = values.get("-CHECKPOINT-", "").strip()
        checkpoint = Path(checkpoint_text) if checkpoint_text else None

        out_folder = imgs_folder / "mae_saliency"
        out_folder.mkdir(exist_ok=True)

        clip_vector = resolve_clip_vector_value(values)

        job = MaeJob(
            imgs_folder=imgs_folder,
            out_folder=out_folder,
            bands=bands,
            checkpoint=checkpoint,
            clip_vector=clip_vector,
            tile_overlap=tile_overlap,
            normalization=normalization,
            compression=resolve_compression_value(values),
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="MAE detection",
        window_title="MAE detection",
    )
    tw.add_rows(
        [sg.Text(
            "Use the Masked Autoencoder technique to detect regions and "
            "images with higher probabilities of containing crop marks."
        )],
        [sg.Text(
            "Extract the MAE saliency values for all the GeoTIFF images "
            "inside the input folder."
        )],
    )
    tw.add_rows(*folder_input_row("-FOLDER-", "Input folder:"))
    tw.add_rows(
        [
            sg.Text("Bands to process (1-based, comma-separated):"),
            sg.Input("1", key="-BANDS-", size=(12, 1)),
        ],
        [sg.Text(
            "Enter one to three bands. One band is repeated across all "
            "three model channels.",
            font=("Arial", 10, "italic"),
        )],
        [sg.Text(
            "Optional local MAE checkpoint "
            "(blank downloads or uses cached weights):"
        )],
        [
            sg.Input(key="-CHECKPOINT-"),
            sg.FileBrowse(
                file_types=(
                    ("Model checkpoints", "*.safetensors;*.bin;*.pth;*.pt"),
                    ("All files", "*.*"),
                )
            ),
        ],
        [
            sg.Text("Tile overlap (px):"),
            sg.Input(str(DEFAULT_TILE_OVERLAP), key="-OVERLAP-", size=(8, 1)),
        ],
        [
            sg.Text("Contrast normalization:"),
            sg.Combo(
                list(_NORMALIZATION_LABELS.keys()),
                key="-NORMALIZATION-",
                default_value="Per-tile (default)",
                readonly=True,
                size=(20, 1),
            ),
        ],
    )
    tw.add_rows(*clip_vector_row())
    tw.add_rows(*compression_row())
    tw.run(job=run_job, start_message="Starting computations...")
