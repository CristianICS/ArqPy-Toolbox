from pathlib import Path

import PySimpleGUI as sg

from sensors import SENSORS
from spectral_indices.job import SpectralIndicesJob

from .framework import (
    ToolWindow,
    clip_vector_row,
    compression_row,
    output_folder_row,
    resolve_clip_vector_value,
    resolve_compression_value,
    resolve_output_folder_value,
    sensor_combo_row,
)


def run_job(values: dict):
    """Compute every spectral index the selected sensor's bands support."""
    try:
        raw_img = values.get("-IMG-", "")
        if not raw_img:
            return ValueError("Please select a valid image.")
        img_path = Path(raw_img)
        if not img_path.exists():
            return FileNotFoundError(str(img_path))

        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: img_path.parent.parent / "SPECTRAL_INDICES",
        )
        clip_vector = resolve_clip_vector_value(values)

        sensor_cls = SENSORS[values["-SENSOR-"]]

        job = SpectralIndicesJob(
            img_path=img_path,
            out_folder=out_folder,
            sensor_band_names=sensor_cls.BAND_KEYS,
            sensor_band_pos=sensor_cls.BAND_POS,
            clip_vector=clip_vector,
            compression=resolve_compression_value(values),
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="Spectral Indices",
        window_title="Spectral Indices",
    )
    tw.add_rows(
        [sg.Text("Input image:")],
        [sg.Input(key="-IMG-"), sg.FileBrowse()],
    )
    tw.add_rows(*output_folder_row("-OUT-", "Select folder to store the results:"))
    tw.add_rows(*clip_vector_row())
    tw.add_rows(*sensor_combo_row())
    tw.add_rows(*compression_row(warning=(
        "Warning: use a Lossless input image. Ratio indices can amplify errors\n"
        "carried from near-lossless inputs in dark pixels, so the output is no\n"
        "longer guaranteed to stay within the error bound."
    )))
    tw.run(job=run_job, start_message="Starting computations...")
