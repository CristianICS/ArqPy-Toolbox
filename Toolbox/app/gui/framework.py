"""Shared PySimpleGUI window scaffolding for the toolbox's tool GUIs.

Every tool window used to duplicate the same ~40 lines: theme, title,
Run/Exit buttons, an Output log box, window construction, the event loop,
`perform_long_operation` dispatch, and the isinstance-based result-popup
chain. `ToolWindow` is the single implementation of that scaffolding;
`clip_vector_row`/`output_folder_row`/`sensor_combo_row` (with their
matching `resolve_*` helpers) are the single implementation of the widget
blocks duplicated across several tools' layouts.

The "catch Exception and return it instead of raising" convention at the
`run_job` boundary is intentionally kept as-is here, since
`perform_long_operation`'s worker-thread return value is how PySimpleGUI
hands the result back to the event loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import PySimpleGUI as sg

from sensors import SENSORS

from .icon_base64 import ICON
from core.clip import resolve_clip_vector
from core.raster_io import LOSSLESS, NEAR_LOSSLESS

# GUI label -> `core.raster_io` compression mode.
COMPRESSION_LABELS = {
    "Lossless": LOSSLESS,
    "Near-lossless": NEAR_LOSSLESS,
}

JobFn = Callable[[dict], object]

# Key of the scrollable column that holds each tool's form rows.
FORM_KEY = "-FORM-"
# Fraction of the screen height the window may take on open, leaving room
# for the title bar and the taskbar.
MAX_SCREEN_FRACTION = 0.85
# The form never shrinks below this height (pixels), so a few rows stay
# visible even on very small screens.
MIN_FORM_HEIGHT = 150


def report_job_result(result: object, success_message: str = "Processing completed successfully.") -> None:
    """
    Single implementation of the popup-dispatch chain to avoid duplicating
    it across every tool GUI.
    
    `RuntimeError` has no separate branch here: it's an `Exception` subclass,
    so the old per-file `isinstance(result, RuntimeError)` branch checked after
    the blanket `Exception` branch could never fire.
    """
    if isinstance(result, FileNotFoundError):
        sg.popup_ok("The input path is not valid.", str(result))
        print(f"Error: {result}")
    elif isinstance(result, Exception):
        sg.popup_ok("An error occurred:", str(result))
        print(f"Error: {result}")
    else:
        sg.popup_ok(success_message)
        print(success_message)


@dataclass
class ToolWindow:
    """Encapsulates the theme/title/Run-Exit/Output-log/window/event-loop/
    perform_long_operation/result-popup boilerplate shared by every tool.

    Usage:
        tw = ToolWindow(title="Atmospheric Correction Tool", window_title="Atmospheric Correction")
        tw.add_rows(*folder_input_row("-IMG-", "Input Image Folder:"))
        tw.add_rows(*output_folder_row("-OUT-"))
        tw.run(job=run_job)
    """
    title: str
    window_title: str
    rows: List[list] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Set the theme as soon as the window is created, not when it's
        # run: `main()` builds its rows via the factory functions below
        # (clip_vector_row, sensor_combo_row, etc.) between constructing
        # this object and calling `run()`, and PySimpleGUI bakes the
        # active theme's colors into each element at construction time.
        # Setting the theme any later leaves those widgets on the default
        # theme instead of VSCodeDark.
        sg.theme("VSCodeDark")

    def add_rows(self, *rows: list) -> "ToolWindow":
        self.rows.extend(rows)
        return self

    def run(self, job: JobFn, *, start_message: str = "Starting processing...") -> None:
        # The form rows live in a scrollable column, so tools with many
        # options still fit on small screens; the title, Run/Exit and the
        # Output log stay fixed outside it. `size_subsample_height=1` sizes
        # the column to its full content; `_fit_to_screen` shrinks it only
        # when the window would be taller than the screen.
        form = sg.Column(
            self.rows,
            key=FORM_KEY,
            scrollable=True,
            vertical_scroll_only=True,
            size_subsample_height=1,
            expand_x=True,
            expand_y=True,
            pad=0,
        )
        layout = [
            [sg.Text(self.title, font=("Arial", 16, "bold"))],
            [form],
            [sg.Button("Run"), sg.Button("Exit")],
            [sg.Output(size=(80, 20), expand_x=True, expand_y=True)],
        ]

        window = sg.Window(
            self.window_title, layout, icon=ICON, resizable=True, finalize=True,
        )
        _fit_to_screen(window, window[FORM_KEY])
        try:
            while True:
                event, values = window.read()

                if event in (sg.WINDOW_CLOSED, "Exit"):
                    break

                if event == "Run":
                    window["Run"].update(disabled=True)
                    print(start_message)
                    window.perform_long_operation(
                        lambda values=values: job(values), "-JOB-DONE-"
                    )

                elif event == "-JOB-DONE-":
                    window["Run"].update(disabled=False)
                    report_job_result(values["-JOB-DONE-"])
        finally:
            window.close()


def _fit_to_screen(window: sg.Window, form: sg.Column) -> None:
    """Shrink the scrollable form so the whole window fits on the screen.
    Only the form loses height; the title, Run/Exit and the Output log keep
    their size. The window stays resizable, so the user can enlarge it again."""
    window.refresh()
    _, screen_h = sg.Window.get_screen_size()
    overflow = window.TKroot.winfo_reqheight() - int(screen_h * MAX_SCREEN_FRACTION)
    if overflow > 0:
        canvas = form.TKColFrame.canvas
        canvas.config(height=max(MIN_FORM_HEIGHT, canvas.winfo_reqheight() - overflow))
        window.refresh()
    window.move_to_center()


def folder_input_row(key: str, label: str) -> list:
    """A labelled folder picker: `sg.Text(label)` + `sg.Input` + `sg.FolderBrowse()`."""
    return [
        [sg.Text(label)],
        [sg.Input(key=key), sg.FolderBrowse()],
    ]


def output_folder_row(key: str = "-OUT-", label: str = "Output Folder:") -> list:
    """The "blank = autocreate" output-folder picker, duplicated (with slight
    wording differences) across several tool GUIs."""
    return [
        [sg.Text(label)],
        [sg.Text("Leave it blank if you want to autocreate it", font=("Arial", 10, "italic"))],
        [sg.Input(key=key), sg.FolderBrowse()],
    ]


def clip_vector_row(key: str = "-CLIP_VECTOR-") -> list:
    """The clip-vector file picker, duplicated across several tool GUIs.
    KML is intentionally excluded from the file-type filter: the toolbox's
    packaged GDAL build has no LIBKML driver, so offering it here would be
    misleading (see the README's "Known limitations" section)."""
    return [
        [sg.Text("Vector layer to perform a clip operation:")],
        [sg.Text("Leave it blank to skip", font=("Arial", 10, "italic"))],
        [
            sg.Input(key=key),
            sg.FileBrowse(file_types=(
                ("Vector files", "*.shp;*.gpkg;*.geojson"),
                ("All files", "*.*"),
            )),
        ],
    ]


def sensor_combo_row(key: str = "-SENSOR-", default: str = "WV3") -> list:
    """Sensor dropdown sourced from the shared `sensors.SENSORS` registry,
    replacing the hardcoded WV3/LEGION dicts duplicated in several GUIs."""
    return [
        [sg.Text("Sensor"), sg.Combo(
            list(SENSORS.keys()),
            key=key,
            default_value=default,
            readonly=True,
            size=(20, 1),
        )],
    ]


def compression_row(key: str = "-COMPRESSION-", warning: Optional[str] = None) -> list:
    """Output-compression dropdown: lossless (bit-exact, the default) or
    near-lossless (LERC with a bounded per-pixel error, much smaller).
    `warning` adds a highlighted line below it, for tools whose outputs feed
    further processing steps."""
    rows = [
        [sg.Text("Output compression"), sg.Combo(
            list(COMPRESSION_LABELS.keys()),
            key=key,
            default_value="Lossless",
            readonly=True,
            size=(20, 1),
        )],
        [sg.Text(
            "Near-lossless keeps float32 but allows a small, bounded error per pixel",
            font=("Arial", 10, "italic"),
        )],
    ]
    if warning:
        rows.append([sg.Text(
            warning, font=("Arial", 10, "bold"), text_color="orange",
        )])
    return rows


def resolve_compression_value(values: dict, key: str = "-COMPRESSION-") -> str:
    """Map the compression dropdown's label to a `core.raster_io` mode;
    a missing/blank value means lossless."""
    return COMPRESSION_LABELS.get(values.get(key) or "Lossless", LOSSLESS)


def resolve_clip_vector_value(values: dict, key: str = "-CLIP_VECTOR-") -> Optional[Path]:
    """Resolve a clip-vector field value via `core.clip.resolve_clip_vector`"""
    return resolve_clip_vector(values.get(key, ""))


def resolve_output_folder_value(
    values: dict,
    key: str,
    default_factory: Callable[[], Path],
    *,
    allow_existing_default: bool = True,
) -> Path:
    """Resolve an output-folder field value: blank means "autocreate the
    default", non-blank must already exist. `allow_existing_default=False`
    raises if the autocreated default already exists, for tools where
    reusing it would silently mix outputs from a previous run; the default
    (True) allows reusing it, for tools designed to resume/skip existing
    outputs on re-run."""
    raw = values.get(key, "")
    if not raw:
        out_folder = default_factory()
        if out_folder.exists() and not allow_existing_default:
            raise FileExistsError(
                f"The default output folder already exists: {out_folder}. "
                "Please choose a different one or remove it before re-running."
            )
        out_folder.mkdir(parents=True, exist_ok=True)
        return out_folder

    out_folder = Path(raw)
    if not out_folder.exists():
        raise NotADirectoryError(f"The output folder does not exist: {out_folder}")
    return out_folder
