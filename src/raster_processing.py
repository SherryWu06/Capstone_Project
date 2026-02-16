"""
GeoTIFF reading and preprocessing for eBird Status and Trends data.
Loads weekly abundance raster stacks for migration analysis.
"""

from pathlib import Path
import json
import numpy as np
import rasterio
from typing import Optional


def find_species_data(data_dir: Path, species: str = "yebsap-example", year: int = 2023) -> Path:
    """Find the species data directory (e.g. data/raw/2023/yebsap-example)."""
    path = data_dir / str(year) / species
    if not path.exists():
        raise FileNotFoundError(f"Species data not found: {path}")
    return path


def load_config(species_dir: Path) -> dict:
    """Load config.json from species directory."""
    config_path = species_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def load_weekly_abundance(
    species_dir: Path,
    product: str = "abundance_median",
    resolution: str = "27km",
    year: int = 2023,
) -> tuple[np.ndarray, dict]:
    """
    Load weekly abundance raster stack.

    Returns:
        data: array of shape (n_weeks, height, width)
        meta: dict with 'dates', 'crs', 'transform', 'bounds'
    """
    species = species_dir.name
    weekly_dir = species_dir / "weekly"
    if not weekly_dir.exists():
        weekly_dir = species_dir  # some layouts put files directly in species dir

    pattern = f"{species}_{product}_{resolution}_{year}.tif"
    tif_path = weekly_dir / pattern
    if not tif_path.exists():
        raise FileNotFoundError(f"Abundance raster not found: {tif_path}")

    config = load_config(species_dir)
    date_names = config.get("DATE_NAMES", [f"{i:02d}" for i in range(1, 53)])

    with rasterio.open(tif_path) as src:
        data = src.read()  # shape: (n_bands, height, width)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "dates": date_names[: data.shape[0]],
            "height": src.height,
            "width": src.width,
        }

    return data, meta


def load_weekly_stack(
    data_dir: Path,
    species: str = "yebsap-example",
    product: str = "abundance_median",
    resolution: str = "27km",
    year: int = 2023,
) -> tuple[np.ndarray, dict]:
    """
    Convenience: load weekly abundance from project data directory.

    Args:
        data_dir: e.g. Path("data/raw")
        species: species code (yebsap-example, woothr, etc.)
        product: abundance_median, abundance_lower, abundance_upper
        resolution: 3km, 9km, or 27km

    Returns:
        data: (n_weeks, height, width)
        meta: metadata dict
    """
    species_dir = find_species_data(data_dir, species, year)
    return load_weekly_abundance(species_dir, product, resolution, year)


def get_season_dates(species_dir: Path) -> list[dict]:
    """Extract season dates from config for labeling."""
    config = load_config(species_dir)
    return config.get("season_dates", [])
