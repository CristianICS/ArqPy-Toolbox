from pathlib import Path

import PySimpleGUI as sg

from sam3_cropmarks import parse_rgb_bands
from sam3_cropmarks.job import Sam3CropmarksJob

from .framework import (
    ToolWindow,
    compression_row,
    output_folder_row,
    resolve_compression_value,
    resolve_output_folder_value,
)


def run_job(values: dict):
    """Segment crop marks in one GeoTIFF or every GeoTIFF in a folder."""
    try:
        raw_input = values.get("-INPUT-", "")
        if not raw_input:
            return ValueError("Please select an input GeoTIFF or folder.")
        input_path = Path(raw_input)
        if not input_path.exists():
            return FileNotFoundError(f"Input path not found: {input_path}")

        input_folder = input_path if input_path.is_dir() else input_path.parent
        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: input_folder / "sam3_crop_marks",
        )

        checkpoint_text = values.get("-CHECKPOINT-", "").strip()
        checkpoint = Path(checkpoint_text) if checkpoint_text else None

        job = Sam3CropmarksJob(
            input_path=input_path,
            output_folder=out_folder,
            prompt=values["-PROMPT-"].strip(),
            confidence=float(values["-CONFIDENCE-"]),
            rgb_bands=parse_rgb_bands(values["-RGB-"]),
            tile_size=int(values["-TILE-SIZE-"]),
            overlap=int(values["-OVERLAP-"]),
            checkpoint=checkpoint,
            overwrite=values["-OVERWRITE-"],
            compression=resolve_compression_value(values),
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="SAM 3 Crop-mark Segmentation",
        window_title="SAM 3 Crop-mark Segmentation",
    )
    tw.add_rows(
        [sg.Text("Extract high-confidence crop-mark masks from one GeoTIFF or a folder.")],
        [sg.Text("Input GeoTIFF or folder:")],
        [
            sg.Input(key="-INPUT-"),
            sg.FileBrowse(
                "Select TIFF",
                target="-INPUT-",
                file_types=(("GeoTIFF images", "*.tif;*.tiff"),),
            ),
            sg.FolderBrowse("Select Folder", target="-INPUT-"),
        ],
    )
    tw.add_rows(*output_folder_row(
        "-OUT-", "Output folder (blank creates 'sam3_crop_marks' beside the input):"
    ))
    tw.add_rows(
        [sg.Text("Text prompt:"), sg.Input("crop mark", key="-PROMPT-", size=(35, 1))],
        [
            sg.Text("Minimum confidence:"),
            sg.Input("0.70", key="-CONFIDENCE-", size=(8, 1)),
            sg.Text("RGB bands (1-based):"),
            sg.Input("1,2,3", key="-RGB-", size=(9, 1)),
        ],
        [
            sg.Text("Tile size:"), sg.Input("1008", key="-TILE-SIZE-", size=(8, 1)),
            sg.Text("Overlap:"), sg.Input("128", key="-OVERLAP-", size=(8, 1)),
        ],
        [sg.Text("Optional local SAM 3 checkpoint (blank uses Hugging Face):")],
        [
            sg.Input(key="-CHECKPOINT-"),
            sg.FileBrowse(file_types=(("PyTorch checkpoint", "*.pt;*.pth"), ("All files", "*.*"))),
        ],
        [sg.Checkbox("Overwrite existing outputs", key="-OVERWRITE-", default=False)],
    )
    tw.add_rows(*compression_row())
    tw.add_rows([sg.Text(
        "Applies to the confidence raster; the crop-mark mask is always lossless",
        font=("Arial", 10, "italic"),
    )])
    tw.run(job=run_job, start_message="Starting crop-mark segmentation...")
