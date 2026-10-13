from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path
import csv

import rasterio
import numpy as np
from rasterio.windows import Window

from core.tiling import iter_blocks

# Side of the windows `compute_raster_stats` reads one at a time, and the
# approximate number of values it keeps for the percentiles.
STATS_BLOCK_SIZE = 1024
STATS_MAX_SAMPLES = 1_000_000

@dataclass(slots=True)
class Stats:
    """
    Store statistics for a derived image product.
    
    Attributes:
    -----------
    id : str
        Derived image product name.
    mean : float
        Mean value of the derived product's saliency map.
    median : float
        Median value of the saliency map.
    min : float
        Minimum value of the saliency map.
    p90 : float
        90th percentile of the saliency map.
    p95 : float
        95th percentile of the saliency map.
    max : float
        Maximum value of the saliency map.
    std : float
        Standard deviation of the saliency map.
    """
    id: str
    mean: float
    median: float
    min: float
    p90: float
    p95: float
    max: float
    std: float

    def round_to2(self) -> "Stats":
        """Return a copy with all statistical values rounded to two decimals."""
        return replace(
            self,
            mean=round(self.mean, 2),
            median=round(self.median, 2),
            min=round(self.min, 2),
            p90=round(self.p90, 2),
            p95=round(self.p95, 2),
            max=round(self.max, 2),
            std=round(self.std, 2),
        )
    
    def to_csv_row(self):
        """Return the stored data as a dictionary suitable for a CSV row."""
        return asdict(self.round_to2())


def is_inside_stats(stats_path: str | Path, img_name: str):
    """Check if a derived image is already inside a CSV stats file."""
    stats_path = Path(stats_path)

    if not stats_path.exists():
        return False

    with stats_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(
                "The provided CSV file with the saliency stats "
                f"does not contain a valid 'id' column: {stats_path}"
            )

        return any(row["id"] == img_name for row in reader)


def update_csv(csv_path: str | Path, stats: Stats):
    """
    Store statistics by image inside a CSV file

    Creates the file if it does not exist.
    Avoids duplicates based on img_name.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    stats : Stats
        Statistics to store.
        
    Returns
    -------
    bool
        True if a row was added; False if the image product
        was already present in the CSV file.
    """
    csv_path = Path(csv_path)

    fieldnames = [field.name for field in fields(Stats)]

    # Do not add duplicate entries
    if (
        csv_path.exists()
        and csv_path.stat().st_size > 0
        and is_inside_stats(csv_path, stats.id)
    ):
        return False

    row = stats.to_csv_row()

    # Write the header when creating a new or empty file
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            writer.writerow(row)
    
    return True


def compute_raster_stats(tif_path: str| Path, band = 1) -> tuple[Stats, np.ndarray]:
    """
    Compute basic statistics for a raster band using rasterio.

    The band is read one window at a time, so it is never held in memory.
    Mean, min, max and std use every valid pixel. Median, p90 and p95 use
    a regular subsample of about `STATS_MAX_SAMPLES` valid pixels (every
    valid pixel when the raster has fewer).

    Parameters
    ----------
    path : str | Path
        Path to the raster file.
    band : int, optional
        Band number to read (default is 1).

    Returns
    -------
    (Stats, np.ndarray)
        The statistics and the subsample of valid values.
    """
    tif_path = Path(tif_path)

    count = 0
    total = 0.0
    total_sq = 0.0
    vmin, vmax = np.inf, -np.inf
    samples = []

    with rasterio.open(tif_path) as src:
        sample_step = max(1, src.width * src.height // STATS_MAX_SAMPLES)
        for block in iter_blocks(
                src.width, src.height, STATS_BLOCK_SIZE, STATS_BLOCK_SIZE):
            # Read as a masked array so nodata values are ignored automatically
            data = src.read(band, masked=True, window=Window(
                block.xoff, block.yoff, block.xsize, block.ysize))
            block_values = data.compressed().astype(np.float64)
            if block_values.size == 0:
                continue
            count += block_values.size
            total += block_values.sum()
            total_sq += np.square(block_values).sum()
            vmin = min(vmin, block_values.min())
            vmax = max(vmax, block_values.max())
            samples.append(block_values[::sample_step])

    # Guard against empty rasters (e.g. all pixels are nodata)
    if count == 0:
        raise ValueError("Raster band contains no valid data.")

    values = np.concatenate(samples)
    mean = total / count
    median, p90, p95 = np.percentile(values, (50, 90, 95))

    stats = Stats(
        id = tif_path.stem,
        mean=mean,
        median=median,
        min=vmin,
        p90=p90,
        p95=p95,
        max=vmax,
        std=np.sqrt(max(total_sq / count - mean ** 2, 0.0))
    )

    return stats, values


def percentile_rank(x, values):
    """
    Compute percentile rank for a value given raster values.

    Percentile rank does not care about the distribution. Interpretation:

    * p = 0.9 -> higher than 90% of pixels
    * p = 0.5 -> median
    * p = 0.01 -> very low value
    """
    percentile = (values <= x).sum() / values.size

    return percentile
