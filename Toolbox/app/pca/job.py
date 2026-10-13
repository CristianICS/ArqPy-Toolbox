"""Orchestrates a full PCA run: resolve/generate band combinations, compute
each one, convert to COG, and optionally clip to a vector cutline.

Ports the per-combination loop that used to live inline in `gui/pca.py`'s
`run_job()` out of the GUI layer (which also fixes a layering violation:
the original called `sg.popup_ok(...)` directly from inside supposedly
headless business logic). Satisfies `core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.clip import clip_raster_to_vector
from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import (
    COMPRESSION_MODES,
    LOSSLESS,
    profile_for,
    temp_tif,
    translate_to_cog,
)

from .utils import (
    NODATA,
    generate_combis,
    load_combis_from_csv,
    run_pca,
    save_combis_to_csv,
)

# OTB DimensionalityReduction writes float32.
_OTB_DTYPE = "float32"


@dataclass(frozen=True)
class PcaResult:
    output_paths: List[Path]


@dataclass
class PcaJob:
    img_path: Path
    out_folder: Path
    sensor_band_names: List[str]
    sensor_band_pos: List[int]
    # Exactly one of: a combinations CSV (the GUI passes the sensor's
    # curated file by default), or the full sweep of every combination.
    combis_csv: Optional[Path] = None
    all_combinations: bool = False
    clip_vector: Optional[Path] = None
    compression: str = LOSSLESS
    # Components kept per combination (fewer if a combination has fewer
    # bands). The first three hold most of the variance and fill an RGB view.
    nbcomp: int = 3

    def validate(self) -> None:
        if not self.img_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.img_path}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.all_combinations and self.combis_csv is not None:
            raise ValueError(
                "Choose either a combinations CSV or all combinations, not both."
            )
        if not self.all_combinations:
            if self.combis_csv is None:
                raise ValueError("No PCA combinations CSV given.")
            if not Path(self.combis_csv).exists():
                raise FileNotFoundError(
                    f"PCA combinations file not found: {self.combis_csv}"
                )
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(f"Unknown compression mode: {self.compression}")
        if self.nbcomp < 1:
            raise ValueError("The number of PCA components must be at least 1.")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> PcaResult:
        if self.all_combinations:
            combis_dict = generate_combis(
                self.sensor_band_names, self.sensor_band_pos)
        else:
            combis_dict = load_combis_from_csv(self.combis_csv)

        self._record_combis(combis_dict)

        return PcaResult(output_paths=self._run_combis(combis_dict, progress))

    def _out_path(self, combi_id: int) -> Path:
        return self.out_folder / f"COMB{combi_id}.tif"

    def _record_combis(self, combis_dict: dict) -> None:
        """Check `combis_dict` against the folder's `band_combinations.csv`
        and add it there.

        An existing `COMB<id>.tif` is skipped on resume, so it must hold the
        same bands as the requested combination `<id>`. The CSV records
        which bands each output holds; if it disagrees (or has no entry for
        an existing output), nothing runs and a ValueError lists the
        clashing ids.
        """
        ledger_path = self.out_folder / "band_combinations.csv"
        recorded = load_combis_from_csv(ledger_path) if ledger_path.exists() else {}

        clashes = []
        for i, item in combis_dict.items():
            if not self._out_path(i).exists():
                continue
            done = recorded.get(i)
            if done is None or done["pos"] != item["pos"]:
                done_bands = "-".join(done["names"]) if done else "unknown bands"
                clashes.append(
                    f"COMB{i}.tif holds {done_bands}, requested {'-'.join(item['names'])}"
                )
        if clashes:
            raise ValueError(
                f"Existing outputs in {self.out_folder} do not match the "
                "requested combinations. Use another output folder, or give "
                "these combinations unused ids:\n" + "\n".join(clashes)
            )

        save_combis_to_csv(dict(sorted({**recorded, **combis_dict}.items())), ledger_path)

    def _run_combis(self, combis_dict: dict, progress: Progress) -> List[Path]:
        # Intermediates stay lossless; only the final COG uses `compression`.
        lossless_profile = profile_for(_OTB_DTYPE, "pca", LOSSLESS)
        final_profile = profile_for(_OTB_DTYPE, "pca", self.compression)

        output_paths: List[Path] = []
        for i, item in combis_dict.items():
            out_stem = f"COMB{i}"
            out_path = self._out_path(i)

            # Skip combinations already computed in a previous run
            # (`_record_combis` checked they hold the same bands). Temp
            # files have random names, so a crashed run leaves nothing that
            # could be mistaken for a finished output.
            if out_path.exists():
                continue
            progress(f"Computing PCA {i}")

            nbcomp = min(self.nbcomp, len(item["pos"]))
            with temp_tif(self.out_folder, f"{out_stem}_temp") as pca_path:
                run_pca(self.img_path, pca_path, item["pos"], ram=2048, nbcomp=nbcomp)

                if self.clip_vector is None:
                    translate_to_cog(
                        pca_path, out_path, final_profile, nodata=NODATA)
                else:
                    with temp_tif(self.out_folder, f"{out_stem}_toclip") as toclip_path:
                        translate_to_cog(
                            pca_path, toclip_path, lossless_profile,
                            nodata=NODATA)
                        clip_raster_to_vector(
                            toclip_path, self.clip_vector, out_path,
                            nodata=NODATA,
                            profile=final_profile,
                            as_cog=True,
                        )
            output_paths.append(out_path)
            progress(f"PCA image saved to: {out_path}")

        return output_paths
