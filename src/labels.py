"""
Map week indices to season labels from eBird config or Matt's JSON.
"""

from datetime import datetime
import json
import numpy as np
from pathlib import Path
from typing import Any


def parse_week_date(date_str: str, year: int = 2023) -> datetime:
    """Parse MM-DD string to datetime."""
    return datetime.strptime(f"{year}-{date_str}", "%Y-%m-%d")


def date_in_season(d: datetime, start: datetime, end: datetime) -> bool:
    """Check if date d falls within [start, end]. Handles year wrap (e.g. Nov-Mar)."""
    if start <= end:
        return start <= d <= end
    return d >= start or d <= end


def build_week_labels(
    date_names: list[str],
    season_dates: list[dict[str, Any]],
    year: int = 2023,
) -> tuple[np.ndarray, list[str]]:
    """
    Map each week to a season label.

    Args:
        date_names: e.g. ["01-04", "01-11", ...] from config DATE_NAMES
        season_dates: list of {"season", "start_date", "end_date"} from config
        year: prediction year

    Returns:
        labels: int array (0..n_classes-1), one per week
        class_names: list of season names in order
    """
    class_names = [s["season"] for s in season_dates]
    n_weeks = len(date_names)
    labels = np.full(n_weeks, -1, dtype=int)

    for i, date_str in enumerate(date_names):
        d = parse_week_date(date_str, year)
        for j, s in enumerate(season_dates):
            start = datetime.strptime(s["start_date"], "%Y-%m-%d")
            end = datetime.strptime(s["end_date"], "%Y-%m-%d")
            if date_in_season(d, start, end):
                labels[i] = j
                break

    # Unlabeled weeks (gaps between seasons) - assign to nearest
    if np.any(labels == -1):
        for i in np.where(labels == -1)[0]:
            # Use next labeled week or prev
            for j in range(i + 1, n_weeks):
                if labels[j] >= 0:
                    labels[i] = labels[j]
                    break
            else:
                for j in range(i - 1, -1, -1):
                    if labels[j] >= 0:
                        labels[i] = labels[j]
                        break

    return labels, class_names


# Season names that indicate movement (migration)
MIGRATION_SEASONS = {"prebreeding_migration", "postbreeding_migration"}


def build_binary_labels(
    date_names: list[str],
    season_dates: list[dict[str, Any]],
    year: int = 2023,
) -> tuple[np.ndarray, list[str]]:
    """
    Map each week to binary: movement (1) vs no movement (0).
    Migration = movement; breeding/nonbreeding = no movement.

    Returns:
        labels: 0 or 1 per week
        class_names: ["no_movement", "movement"]
    """
    labels_4, _ = build_week_labels(date_names, season_dates, year)
    class_names = ["no_movement", "movement"]

    # Map: migration -> 1, breeding/nonbreeding -> 0
    season_to_binary = {}
    for j, s in enumerate(season_dates):
        season_to_binary[j] = 1 if s["season"] in MIGRATION_SEASONS else 0

    binary = np.array([season_to_binary.get(l, 0) for l in labels_4])
    return binary, class_names


def get_season_dates_from_json(
    labels_path: Path,
    species: str,
    year: int = None,
) -> tuple[list[dict], list[str]]:
    """
    Load season_dates and DATE_NAMES for a species from matt_species_seasons.json.
    Season dates are kept as-is (year from the JSON is used when year=None).

    Returns:
        season_dates: list of {"season", "start_date", "end_date"}
        date_names: list of MM-DD strings
    """
    with open(labels_path) as f:
        data = json.load(f)
    entry = data.get(species)
    if not entry:
        raise KeyError(f"Species {species} not in {labels_path}")

    # Use the year from the JSON unless explicitly overridden
    source_year = entry.get("year", 2024) if year is None else year

    season_dates = []
    for s in entry["season_dates"]:
        start = s["start_date"]  # e.g. "2023-05-17"
        end = s["end_date"]
        start_parts = start.split("-")
        end_parts = end.split("-")
        start_norm = f"{source_year}-{start_parts[1]}-{start_parts[2]}"
        end_norm = f"{source_year}-{end_parts[1]}-{end_parts[2]}"
        season_dates.append({"season": s["season"], "start_date": start_norm, "end_date": end_norm})

    return season_dates, entry["DATE_NAMES"]
