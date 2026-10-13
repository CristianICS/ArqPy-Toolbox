from pathlib import Path

import PySimpleGUI as sg

from pansharp import SENSORS
from pansharp.job import METHODS, PansharpJob

from .framework import (
    ToolWindow,
    clip_vector_row,
    compression_row,
    output_folder_row,
    resolve_clip_vector_value,
    resolve_compression_value,
    resolve_output_folder_value,
)


def run_job(values: dict):
    """Perform a pansharpening operation."""
    try:
        raw_mul = values.get("-MUL-", "")
        if not raw_mul:
            return ValueError("Please select a valid MUL image.")
        mul_img_path = Path(raw_mul)
        if not mul_img_path.exists():
            return FileNotFoundError(f"MUL image not found: {mul_img_path}")

        raw_pan = values.get("-PAN-", "")
        if not raw_pan:
            return ValueError("Please select a valid PAN image.")
        pan_img_path = Path(raw_pan)
        if not pan_img_path.exists():
            return FileNotFoundError(f"PAN image not found: {pan_img_path}")

        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: mul_img_path.parent,
        )
        clip_vector = resolve_clip_vector_value(values)

        try:
            bayes_lambda = float(values["-LAMBDA-"])
        except ValueError:
            return ValueError(
                "Only float numbers between 0 and 1 are allowed for the "
                "Bayes' lambda parameter."
            )

        job = PansharpJob(
            mul_path=mul_img_path,
            pan_path=pan_img_path,
            out_folder=out_folder,
            method=values["-ALGORITHM-"],
            sensor_name=values["-SENSOR-"],
            bayes_lambda=bayes_lambda,
            clip_vector=clip_vector,
            compression=resolve_compression_value(values),
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="Pansharpening Tool",
        window_title="Pansharpening",
    )
    tw.add_rows(
        [sg.Text("Input MUL Image:")],
        [sg.Input(key="-MUL-"), sg.FileBrowse()],
        [sg.Text("Input PAN Image:")],
        [sg.Input(key="-PAN-"), sg.FileBrowse()],
    )
    tw.add_rows(*output_folder_row("-OUT-"))
    tw.add_rows(*clip_vector_row())
    tw.add_rows(*compression_row(warning=(
        "Warning: if this output feeds PCA, spectral indices or MAE, keep it\n"
        "Lossless. Near-lossless errors are amplified by later steps, so the\n"
        "final product is no longer guaranteed to stay within the error bound."
    )))
    tw.add_rows(
        [sg.Text("Sensor"), sg.Combo(
            list(SENSORS.keys()),
            key="-SENSOR-",
            default_value=list(SENSORS.keys())[0],
            readonly=True,
            size=(20, 1)
        )],
        [sg.Text("Algorithm"), sg.Combo(
            list(METHODS),
            key="-ALGORITHM-",
            default_value="Bayesian",
            readonly=True,
            size=(20, 1)
        )],
        [sg.Text("If Bayesian algorithm is selected, define a valid 'lambda' parameter.")],
        [sg.Text(
            "For integer reflectance values (constant applied) use a higher value than for original reflectance (0-1) values",
            font=("Arial", 9, "italic")
        )],
        [sg.Input(default_text='0.995', key='-LAMBDA-')],
    )
    tw.run(job=run_job, start_message="Starting pansharpening operation...")
