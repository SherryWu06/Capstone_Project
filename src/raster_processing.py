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


# Standard 52-week date names (MM-DD) when no config
DEFAULT_DATE_NAMES = [
    "01-04", "01-11", "01-18", "01-25", "02-01", "02-08", "02-15", "02-22",
    "03-01", "03-08", "03-15", "03-22", "03-29", "04-05", "04-12", "04-19", "04-26",
    "05-03", "05-10", "05-17", "05-24", "05-31", "06-07", "06-14", "06-21", "06-28",
    "07-05", "07-12", "07-19", "07-26", "08-02", "08-09", "08-16", "08-23", "08-30",
    "09-06", "09-13", "09-20", "09-27", "10-04", "10-11", "10-18", "10-25", "11-01",
    "11-08", "11-15", "11-22", "11-29", "12-06", "12-13", "12-20", "12-27",
]


def load_matt_stack(
    data_dir: Path,
    species: str,
    product: str = "abundance_median",
    resolution: str = "27km",
    year: int = None,
) -> tuple[np.ndarray, dict]:
    """
    Load weekly abundance from Matt's flat layout (data/raw/Matt/).

    Args:
        data_dir: e.g. Path("data/raw")
        species: species code (acafly, etc.)
        product: abundance_median (prefer), or occurrence_median if available
        resolution: 3km, 9km, or 27km
        year: optional year filter; if None, uses the most recent available file

    Returns:
        data: (n_weeks, height, width)
        meta: dict with dates, crs, transform, etc.
    """
    matt_dir = data_dir / "Matt"
    if not matt_dir.exists():
        raise FileNotFoundError(f"Matt data directory not found: {matt_dir}")

    if year is not None:
        matches = sorted(matt_dir.glob(f"{species}_{product}_{resolution}_{year}.tif"))
    else:
        matches = sorted(matt_dir.glob(f"{species}_{product}_{resolution}_*.tif"))

    if not matches:
        raise FileNotFoundError(
            f"No raster found for {species} ({product}, {resolution}) in {matt_dir}"
        )
    if len(matches) > 1:
        import warnings
        warnings.warn(
            f"Multiple files found for {species} ({product}, {resolution}): "
            f"{[m.name for m in matches]}. Using most recent: {matches[-1].name}",
            stacklevel=2,
        )
    tif_path = matches[-1]

    with rasterio.open(tif_path) as src:
        data = src.read()  # (n_bands, height, width)
        n_bands = data.shape[0]
        date_names = DEFAULT_DATE_NAMES[:n_bands]
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "dates": date_names,
            "height": src.height,
            "width": src.width,
        }

    return data, meta


def list_ebirdst_species(
    data_dir: Path,
    product: str = "abundance_median",
    resolution: str = "27km",
    year: int = 2023,
) -> list[str]:
    """
    Discover species with weekly TIF files in data/raw/{year}/{species}/.
    Returns species codes that have abundance_median weekly rasters.
    """
    year_dir = data_dir / str(year)
    if not year_dir.exists():
        return []
    species_list = []
    for species_dir in sorted(year_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        weekly_dir = species_dir / "weekly"
        if not weekly_dir.exists():
            weekly_dir = species_dir
        tif_path = weekly_dir / f"{species_dir.name}_{product}_{resolution}_{year}.tif"
        if tif_path.exists():
            species_list.append(species_dir.name)
    return species_list


def list_matt_species(data_dir: Path, product: str = "abundance_median", year: int = None) -> list[str]:
    """
    Discover species with TIF files in data/raw/Matt/.
    Returns species codes that have abundance_median files.
    If year is given, only returns species with that year. Otherwise returns all.
    """
    matt_dir = data_dir / "Matt"
    if not matt_dir.exists():
        return []
    species = set()
    glob_pattern = f"*_{product}_*_{year}.tif" if year else f"*_{product}_*.tif"
    for f in matt_dir.glob(glob_pattern):
        # Pattern: {species}_{product}_{resolution}_{year}.tif
        parts = f.stem.split("_")
        if len(parts) >= 4:
            species.add(parts[0])
    return sorted(species)
