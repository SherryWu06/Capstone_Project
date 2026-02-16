"""
Map week indices to season labels from eBird config.
"""

from datetime import datetime
import numpy as np
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
