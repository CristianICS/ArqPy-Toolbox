from pathlib import Path

import PySimpleGUI as sg

from highpass.job import HighPassJob

from .framework import (
    ToolWindow,
    clip_vector_row,
    compression_row,
    output_folder_row,
    resolve_clip_vector_value,
    resolve_compression_value,
    resolve_output_folder_value,
)


def _parse_band(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError("The band index must be an integer number.")


def run_job(values: dict):
    """Compute high-pass filters over one image band."""
    try:
        raw_img = values.get("-IMG-", "")
        if not raw_img:
            return ValueError("Please select a valid image.")
        img_path = Path(raw_img)
        if not img_path.exists():
            return FileNotFoundError(str(img_path))

        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: img_path.parent.parent / "HIGHPASS",
        )
        clip_vector = resolve_clip_vector_value(values)
        band = _parse_band(values["-BAND-"])

        job = HighPassJob(
            img_path=img_path,
            out_folder=out_folder,
            band=band,
            clip_vector=clip_vector,
            compression=resolve_compression_value(values),
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="Highpass filters",
        window_title="Highpass filters",
    )
    tw.add_rows(
        [sg.Text(
            "Apply common high-pass filters (Laplacian, LoG, Sobel "
            "magnitude, High-boost) using GDAL+NumPy."
        )],
        [sg.Text("Input image:")],
        [sg.Input(key="-IMG-"), sg.FileBrowse()],
        [sg.Text("Image band to process:")],
        [sg.Text("Band index (1-based). Default: 1", font=("Arial", 10, "italic"))],
        [sg.Input(key="-BAND-", default_text="1")],
    )
    tw.add_rows(*output_folder_row("-OUT-"))
    tw.add_rows(*clip_vector_row())
    tw.add_rows(*compression_row())
    tw.run(job=run_job, start_message="Starting computations...")
