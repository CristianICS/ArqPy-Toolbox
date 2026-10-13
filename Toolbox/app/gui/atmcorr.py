from pathlib import Path

import PySimpleGUI as sg

from atmo_correction import FORMULAS
from atmo_correction.job import AtmoCorrectionJob
from sensors import SENSORS

from .framework import (
    ToolWindow,
    clip_vector_row,
    compression_row,
    folder_input_row,
    output_folder_row,
    resolve_clip_vector_value,
    resolve_compression_value,
    resolve_output_folder_value,
    sensor_combo_row,
)


def _parse_scale(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError("The provided scale value must be an integer number.")


def run_job(values: dict):
    """Perform atmospheric correction on every image in the selected folder."""
    try:
        raw_img = values.get("-IMG-", "")
        if not raw_img:
            return ValueError("Please select a valid image directory.")
        img_dir = Path(raw_img)
        if not img_dir.exists():
            return FileNotFoundError(f"Image folder not found: {img_dir}")

        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: img_dir / (img_dir.name + "_BOA"),
            allow_existing_default=True,
        )
        clip_vector = resolve_clip_vector_value(values)
        scale = _parse_scale(values["-SCALE-"])

        job = AtmoCorrectionJob(
            image_dir=img_dir,
            out_folder=out_folder,
            sensor_cls=SENSORS[values["-SENSOR-"]],
            sensor_name=values["-SENSOR-"],
            atm_formula=values["-ATM-"],
            scale=scale,
            clip_vector=clip_vector,
            compression=resolve_compression_value(values),
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="Atmospheric Correction Tool",
        window_title="Atmospheric Correction",
    )
    tw.add_rows(*folder_input_row("-IMG-", "Input Image Folder:"))
    tw.add_rows(*output_folder_row("-OUT-"))
    tw.add_rows(*clip_vector_row())
    tw.add_rows(*sensor_combo_row())
    tw.add_rows(
        [sg.Text("Atmospheric Correction formula:"), sg.Combo(
            list(FORMULAS.keys()),
            key="-ATM-",
            default_value="DOS3",
            readonly=True,
            size=(20, 1),
        )],
        [sg.Text("Scale")],
        [sg.Text(
            "Apply a constant to export image with integer values",
            font=("Arial", 10, "italic"),
        )],
        [sg.Text("Tap 0 to skip.", font=("Arial", 10, "italic"))],
        [sg.Input(default_text='0', key='-SCALE-')],
    )
    # Near-lossless only applies with Scale = 0 (float32 reflectance);
    # scaled int16 outputs are always lossless.
    tw.add_rows(*compression_row(warning=(
        "Warning: if this output feeds pansharpening or other tools, keep it\n"
        "Lossless. Near-lossless errors are amplified by later steps, so the\n"
        "final product is no longer guaranteed to stay within the error bound."
    )))
    tw.run(job=run_job, start_message="Starting atmospheric correction...")
