"""
Per-cell migration analysis: where and when does movement start?

This is a separate analysis from MIL attention maps. Instead of asking
"which cells help classify a week as migration?", it asks:
  - "Where is movement occurring this week?" (per-week movement maps)
  - "Which week does each cell first show movement?" (onset map)

Approach:
  - Pixel-level: week-to-week absolute change in abundance (direct signal)
  - Cell-level onset: z-score of change per cell; onset = first week above threshold
  - Onset map: each cell colored by its onset week number

Run from project root:
    python scratch/plot_migration_onset.py --species acafly --basemap --region north_america
    python scratch/plot_migration_onset.py --species acafly --weekly --basemap --region north_america
    python scratch/plot_migration_onset.py --species acafly --onset --basemap --region north_america
"""

import argparse
import json
import warnings
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import reproject, Resampling
import geopandas as gpd

warnings.filterwarnings("ignore", category=RuntimeWarning)

_BORDERS_CACHE: dict = {}


def _load_borders():
    """Load and cache Natural Earth country + state boundaries (WGS84)."""
    if "countries" not in _BORDERS_CACHE:
        _BORDERS_CACHE["countries"] = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/50m_cultural/"
            "ne_50m_admin_0_countries.zip"
        )
        _BORDERS_CACHE["states"] = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/50m_cultural/"
            "ne_50m_admin_1_states_provinces.zip"
        )
    return _BORDERS_CACHE["countries"], _BORDERS_CACHE["states"]

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.raster_processing import load_matt_stack, load_weekly_stack

# Region bounds (lon_min, lon_max, lat_min, lat_max) in WGS84
REGION_BOUNDS = {
    "north_america": (-170, -50, 15, 72),
    "americas": (-170, -35, -55, 72),
    "lower_48": (-130, -60, 18, 58),
    "lower_48_plus": (-172, -50, 14, 73),
}

DEFAULT_DATE_NAMES = [
    "01-04", "01-11", "01-18", "01-25", "02-01", "02-08", "02-15", "02-22",
    "03-01", "03-08", "03-15", "03-22", "03-29", "04-05", "04-12", "04-19", "04-26",
    "05-03", "05-10", "05-17", "05-24", "05-31", "06-07", "06-14", "06-21", "06-28",
    "07-05", "07-12", "07-19", "07-26", "08-02", "08-09", "08-16", "08-23", "08-30",
    "09-06", "09-13", "09-20", "09-27", "10-04", "10-11", "10-18", "10-25", "11-01",
    "11-08", "11-15", "11-22", "11-29", "12-06", "12-13", "12-20", "12-27",
]

SEASON_BUFFER_WEEKS = 3


def date_to_week_index(date_str: str, date_names: list = None) -> int | None:
    """Convert an ISO date string (YYYY-MM-DD) to a week index in date_names."""
    if date_names is None:
        date_names = DEFAULT_DATE_NAMES
    try:
        from datetime import datetime
        md = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m-%d")
        for i, d in enumerate(date_names):
            if d == md:
                return i
        target = datetime.strptime(date_str, "%Y-%m-%d")
        best_i, best_diff = 0, 999
        for i, d in enumerate(date_names):
            dt = datetime.strptime(f"2023-{d}", "%Y-%m-%d")
            diff = abs((target - dt).days)
            if diff < best_diff:
                best_diff = diff
                best_i = i
        return best_i
    except (ValueError, TypeError):
        return None


def get_species_search_windows(
    species_data: dict, date_names: list, buffer_weeks: int = SEASON_BUFFER_WEEKS
) -> dict | None:
    """
    Derive per-species onset search windows from eBird season dates,
    with buffer_weeks padding on each side to allow detection
    of early-shifting migration.

    Returns dict with spring_start, spring_end, fall_start, fall_end
    as week indices, or None if season dates are unavailable.
    """
    season_dates = species_data.get("season_dates")
    if not season_dates:
        return None

    seasons = {s["season"]: s for s in season_dates}
    pre = seasons.get("prebreeding_migration")
    post = seasons.get("postbreeding_migration")
    if not pre or not post:
        return None

    pre_start = date_to_week_index(pre["start_date"], date_names)
    pre_end = date_to_week_index(pre["end_date"], date_names)
    post_start = date_to_week_index(post["start_date"], date_names)
    post_end = date_to_week_index(post["end_date"], date_names)

    if any(v is None for v in (pre_start, pre_end, post_start, post_end)):
        return None

    n_weeks = len(date_names)
    return {
        "spring_start": max(0, pre_start - buffer_weeks),
        "spring_end": min(n_weeks, pre_end + buffer_weeks + 1),
        "fall_start": max(0, post_start - buffer_weeks),
        "fall_end": min(n_weeks, post_end + buffer_weeks + 1),
    }


def reproject_to_lonlat(
    array: np.ndarray,
    src_crs,
    src_transform,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    dst_width: int = 1800,
) -> tuple[np.ndarray, list]:
    """
    Reproject a 2-D array from its native CRS to WGS84 lon/lat,
    clipped to the given geographic bounding box.

    Returns (reprojected_array, extent) where extent = [lon_min, lon_max, lat_min, lat_max].
    """
    dst_height = int(dst_width * (lat_max - lat_min) / (lon_max - lon_min))
    dst_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, dst_width, dst_height)
    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    reproject(
        source=array.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    extent = [lon_min, lon_max, lat_min, lat_max]
    return dst, extent


def add_basemap(ax, lon_min, lon_max, lat_min, lat_max):
    """Draw country borders and state/province boundaries from Natural Earth."""
    countries, states = _load_borders()

    countries.boundary.plot(ax=ax, linewidth=0.4, edgecolor="#555555", zorder=2)
    states.boundary.plot(ax=ax, linewidth=0.2, edgecolor="#aaaaaa", zorder=2)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")


def compute_weekly_change(stack: np.ndarray) -> np.ndarray:
    """
    Pixel-level absolute change between consecutive weeks.
    Returns array of shape (n_weeks, height, width).
    Week 0 = 0 (no previous week).
    """
    n_weeks, h, w = stack.shape
    change = np.zeros((n_weeks, h, w), dtype=np.float32)
    s = stack.astype(np.float64)
    for t in range(1, n_weeks):
        diff = np.abs(s[t] - s[t - 1])
        diff[~np.isfinite(diff)] = 0
        change[t] = diff
    return change


def compute_cell_onset(
    change: np.ndarray,
    cell_size: int = 16,
    z_threshold: float = 1.5,
    search_start: int = 5,
    search_end: int = 51,
) -> np.ndarray:
    """
    For each cell, find the first week (within search_start..search_end)
    where mean change exceeds z_threshold standard deviations above the cell mean.

    Returns onset map (n_rows, n_cols), value = onset week index, or NaN if not found.
    """
    n_weeks, h, w = change.shape
    n_rows = h // cell_size
    n_cols = w // cell_size

    if n_rows == 0 or n_cols == 0:
        return np.full((max(1, n_rows), max(1, n_cols)), np.nan, dtype=np.float32)

    h_trim = n_rows * cell_size
    w_trim = n_cols * cell_size

    # Compute cell-level mean change per week via reshape, one week at a time
    # to keep memory manageable for large rasters (e.g. 3km at 5562x11484).
    cell_means = np.empty((n_weeks, n_rows, n_cols), dtype=np.float32)
    for t in range(n_weeks):
        week_slice = np.ascontiguousarray(change[t, :h_trim, :w_trim])
        reshaped = week_slice.reshape(n_rows, cell_size, n_cols, cell_size)
        cell_means[t] = np.nanmean(reshaped, axis=(1, 3))

    mu = np.nanmean(cell_means, axis=0)
    sigma = np.nanstd(cell_means, axis=0)

    # Mask cells with no variation (ocean / constant abundance)
    valid = sigma >= 1e-9
    sigma_safe = np.where(valid, sigma, np.nan)
    z = (cell_means - mu) / sigma_safe

    # Find first week in search window where z >= threshold
    search_z = z[search_start:search_end]
    above = search_z >= z_threshold
    has_onset = np.any(above, axis=0)
    first_idx = np.argmax(above, axis=0)

    onset = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    mask = has_onset & valid
    onset[mask] = (first_idx[mask] + search_start).astype(np.float32)

    return onset


def plot_weekly_movement_maps(
    output_dir: Path,
    species: str,
    stack: np.ndarray,
    date_names: list,
    meta: dict,
    change: np.ndarray,
    use_basemap: bool = False,
    region: str = "full",
    weeks: list | None = None,
) -> None:
    """Save one PNG per week showing pixel-level movement magnitude."""
    src_crs = meta["crs"]
    src_transform = meta["transform"]

    lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS.get(
        region, (-180, 180, -60, 85)
    )

    weekly_dir = output_dir / "movement" / species
    weekly_dir.mkdir(parents=True, exist_ok=True)

    # Reproject mean abundance for background
    s = stack.astype(np.float64)
    s[~np.isfinite(s)] = np.nan
    mean_ab = np.nanmean(s, axis=0)
    ab_norm = np.nan_to_num(mean_ab, nan=0)
    p95 = np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9
    ab_norm = np.clip(ab_norm / p95, 0, 1)
    ab_ll, img_ext = reproject_to_lonlat(
        ab_norm, src_crs, src_transform,
        lon_min, lon_max, lat_min, lat_max,
    )

    # Color scale: 99th percentile of all change values
    all_chg = change[1:]
    vmax = float(np.nanpercentile(all_chg[all_chg > 0], 99)) if np.any(all_chg > 0) else 1.0

    imshow_kw = {"extent": img_ext, "origin": "upper"}
    n_weeks = change.shape[0]
    week_iter = weeks if weeks is not None else range(1, n_weeks)

    for t in week_iter:
        if t >= n_weeks:
            continue
        date_str = date_names[t] if t < len(date_names) else f"W{t:02d}"
        chg_ll, _ = reproject_to_lonlat(
            change[t], src_crs, src_transform,
            lon_min, lon_max, lat_min, lat_max,
        )

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        if use_basemap:
            add_basemap(ax, lon_min, lon_max, lat_min, lat_max)

        ax.imshow(ab_ll, cmap="Greys", alpha=0.6, **imshow_kw)
        im = ax.imshow(
            chg_ll, cmap="hot_r", alpha=0.8,
            norm=mcolors.Normalize(vmin=0, vmax=vmax),
            **imshow_kw,
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(f"{species.upper()} – Movement week of {date_str}", fontsize=14)
        plt.colorbar(im, ax=ax, label="Abundance change (week-to-week)", shrink=0.8)
        plt.tight_layout()

        out_path = weekly_dir / f"{species}_movement_week{t:02d}_{date_str.replace('-', '')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved movement maps for {len(week_iter) if isinstance(week_iter, list) else n_weeks - 1} weeks to {weekly_dir}")


def get_data_bbox(mask, pad=8):
    """Return (r0, r1, c0, c1) bounding box of True values in mask, with padding."""
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0 = max(0, rows.min() - pad)
    r1 = min(mask.shape[0], rows.max() + pad + 1)
    c0 = max(0, cols.min() - pad)
    c1 = min(mask.shape[1], cols.max() + pad + 1)
    return r0, r1, c0, c1


def prepare_onset_map_layers(
    onset: np.ndarray,
    stack: np.ndarray,
    date_names: list,
    meta: dict,
    region: str = "full",
    cell_size: int = 16,
    search_start: int = 0,
    search_end: int = 51,
    display_buffer: int = 0,
    cap_weeks: int | None = None,
) -> dict | None:
    """
    Build lon/lat onset and abundance layers (same preprocessing as plot_onset_map).

    Returns None if there is no valid onset anywhere; otherwise a dict with:
      onset_ll, ab_ll, img_ext, lon_min, lon_max, lat_min, lat_max,
      week_min, week_max, week_min_display, week_max_display,
      n_shown, n_total, date_start_label, date_end_label
    """
    h, w = meta["height"], meta["width"]
    src_crs = meta["crs"]
    src_transform = meta["transform"]

    lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS.get(
        region, (-180, 180, -60, 85)
    )

    valid_weeks = onset[np.isfinite(onset)]
    if len(valid_weeks) == 0:
        return None

    week_min = int(valid_weeks.min())
    week_max = int(valid_weeks.max())

    onset_plot = onset
    if cap_weeks is not None:
        cap_limit = week_min + cap_weeks - 1
        onset_plot = onset.copy()
        onset_plot[np.isfinite(onset_plot) & (onset_plot > cap_limit)] = np.nan
        week_max = min(week_max, cap_limit)

    display_start = max(0, search_start - display_buffer)
    week_min_display = min(week_min, display_start)
    week_max_display = week_max

    n_shown = int(np.sum(np.isfinite(onset_plot)))
    n_total = int(np.sum(np.isfinite(onset)))
    if cap_weeks is not None:
        print(f"  Displaying {n_shown}/{n_total} cells (capped to first {cap_weeks} weeks: "
              f"{week_min}–{week_max}, "
              f"{date_names[week_min] if week_min < len(date_names) else week_min}–"
              f"{date_names[min(week_max, len(date_names)-1)]})")
    else:
        print(f"  Displaying {n_total} cells with onset in weeks {week_min}–{week_max} "
              f"({date_names[week_min] if week_min < len(date_names) else week_min}–"
              f"{date_names[min(week_max, len(date_names)-1)]})")

    onset_up = np.kron(onset_plot, np.ones((cell_size, cell_size)))
    onset_up = onset_up[:h, :w]
    onset_up[onset_up == 0] = np.nan

    s = stack.astype(np.float64)
    s[~np.isfinite(s)] = np.nan
    mean_ab = np.nanmean(s, axis=0)
    ab_norm = np.nan_to_num(mean_ab, nan=0)
    p95 = np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9
    ab_norm = np.clip(ab_norm / p95, 0, 1)

    onset_ll, img_ext = reproject_to_lonlat(
        onset_up, src_crs, src_transform,
        lon_min, lon_max, lat_min, lat_max,
    )
    ab_ll, _ = reproject_to_lonlat(
        ab_norm, src_crs, src_transform,
        lon_min, lon_max, lat_min, lat_max,
    )
    onset_ll[onset_ll == 0] = np.nan

    date_start_label = date_names[week_min] if week_min < len(date_names) else f"W{week_min}"
    date_end_label = date_names[min(week_max, len(date_names) - 1)]

    return {
        "onset_ll": onset_ll,
        "ab_ll": ab_ll,
        "img_ext": img_ext,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "week_min": week_min,
        "week_max": week_max,
        "week_min_display": week_min_display,
        "week_max_display": week_max_display,
        "n_shown": n_shown,
        "n_total": n_total,
        "date_start_label": date_start_label,
        "date_end_label": date_end_label,
    }


def plot_onset_map(
    output_dir: Path,
    species: str,
    onset: np.ndarray,
    stack: np.ndarray,
    date_names: list,
    meta: dict,
    use_basemap: bool = False,
    region: str = "full",
    cell_size: int = 16,
    season: str = "both",
    search_start: int = 0,
    search_end: int = 51,
    display_buffer: int = 0,
    cap_weeks: int | None = None,
    clean: bool = False,
) -> None:
    """
    Plot onset week per cell as a spatial map.
    Each cell is colored by the week number when it first showed movement.

    All detected onset cells within the search window are displayed.
    display_buffer extends the display range before search_start for context.
    cap_weeks, if set, limits display to the first N weeks from the earliest
    detected onset — useful for focusing on where migration begins.
    """
    layers = prepare_onset_map_layers(
        onset=onset,
        stack=stack,
        date_names=date_names,
        meta=meta,
        region=region,
        cell_size=cell_size,
        search_start=search_start,
        search_end=search_end,
        display_buffer=display_buffer,
        cap_weeks=cap_weeks,
    )
    if layers is None:
        print(f"  No onset found for {species}, skipping onset map.")
        return

    onset_ll = layers["onset_ll"]
    ab_ll = layers["ab_ll"]
    img_ext = layers["img_ext"]
    lon_min = layers["lon_min"]
    lon_max = layers["lon_max"]
    lat_min = layers["lat_min"]
    lat_max = layers["lat_max"]
    week_min = layers["week_min"]
    week_max = layers["week_max"]
    week_min_display = layers["week_min_display"]
    week_max_display = layers["week_max_display"]
    date_start_label = layers["date_start_label"]
    date_end_label = layers["date_end_label"]

    # ----- Plot in plain lon/lat -----
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    if use_basemap:
        add_basemap(ax, lon_min, lon_max, lat_min, lat_max)

    imshow_kw = {"extent": img_ext, "origin": "upper"}
    ax.imshow(ab_ll, cmap="Greys", alpha=0.18, **imshow_kw)

    # Discrete colormap: turbo (blue=earliest → red=latest)
    # Turbo has high perceptual contrast between adjacent steps, making each
    # onset week visually distinct — unlike warm ramps where orange/yellow blend.
    weeks_shown = np.arange(week_min_display, week_max_display + 1)
    n = len(weeks_shown)
    turbo = plt.get_cmap("turbo")
    colors = [mcolors.to_hex(turbo(i / max(n - 1, 1))) for i in range(n)]
    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad((1, 1, 1, 0))
    bounds_norm = np.arange(week_min_display - 0.5, week_max_display + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds_norm, cmap.N)

    im = ax.imshow(
        onset_ll,
        cmap=cmap,
        norm=norm,
        alpha=0.95,
        interpolation="nearest",
        **imshow_kw,
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")

    # Colorbar with date labels (subsample ticks if many weeks)
    cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    max_ticks = 10
    if len(weeks_shown) <= max_ticks:
        tick_weeks = weeks_shown
    else:
        tick_idx = np.linspace(0, len(weeks_shown) - 1, max_ticks, dtype=int)
        tick_weeks = weeks_shown[tick_idx]
    cbar.set_ticks(tick_weeks)
    cbar.set_ticklabels([
        date_names[wk] if wk < len(date_names) else f"W{wk}" for wk in tick_weeks
    ])
    cbar.set_label("Migration onset week", fontsize=11)

    if clean:
        ax.set_title("")
        try:
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception:
            pass
    else:
        date_start_label = date_names[week_min] if week_min < len(date_names) else f"W{week_min}"
        date_end_label = date_names[min(week_max, len(date_names) - 1)]
        ax.set_title(
            f"{species.upper()} – Migration onset by region ({season})\n"
            f"(Blue = earliest, Red = latest; {date_start_label}–{date_end_label})",
            fontsize=13,
        )
    plt.tight_layout()

    out_path = output_dir / f"{species}_migration_onset.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved onset map to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-cell migration onset analysis")
    parser.add_argument("--species", nargs="+", default=["acafly", "comyel", "casvir"])
    parser.add_argument("--resolution", default="27km")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--cell-size", type=int, default=16, help="Cell size for onset detection (default: 16)")
    parser.add_argument(
        "--output-dir", default="outputs/presentation",
        help="Output directory (default: outputs/presentation)"
    )
    parser.add_argument("--basemap", action="store_true", help="Add coastlines and borders")
    parser.add_argument(
        "--region",
        choices=["full", "north_america", "americas", "lower_48", "lower_48_plus"],
        default="lower_48",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Generate per-week pixel-level movement maps",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="*",
        default=None,
        help="Limit weekly maps to these week indices (e.g. --weeks 20 21 22)",
    )
    parser.add_argument(
        "--onset",
        action="store_true",
        help="Generate migration onset map (first week of movement per cell)",
    )
    parser.add_argument(
        "--onset-spring-start", type=int, default=None,
        help="Override spring onset search start week (default: auto from eBird season dates, fallback 5)",
    )
    parser.add_argument(
        "--onset-spring-end", type=int, default=None,
        help="Override spring onset search end week (default: auto from eBird season dates, fallback 30)",
    )
    parser.add_argument(
        "--onset-fall-start", type=int, default=None,
        help="Override fall onset search start week (default: auto from eBird season dates, fallback 30)",
    )
    parser.add_argument(
        "--onset-fall-end", type=int, default=None,
        help="Override fall onset search end week (default: auto from eBird season dates, fallback 50)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=1.5,
        help="Z-score threshold for onset detection (default: 1.5)",
    )
    parser.add_argument(
        "--display-weeks", type=int, default=0,
        help="Extra weeks of buffer before the search window start for display context (default: 0)",
    )
    parser.add_argument(
        "--cap-weeks", type=int, default=None,
        help="Show only the first N weeks of onset from the earliest detection (focus on migration front)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove title and axis labels for presentation-ready maps",
    )
    parser.add_argument(
        "--season-buffer", type=int, default=SEASON_BUFFER_WEEKS,
        help=f"Weeks of padding before/after eBird season dates for search window (default: {SEASON_BUFFER_WEEKS})",
    )
    args = parser.parse_args()

    if not args.weekly and not args.onset:
        print("No output mode selected. Use --weekly and/or --onset.")
        parser.print_help()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data" / "raw"
    labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"

    for species in args.species:
        print(f"\nProcessing {species}...")
        try:
            # Try the 2023-layout first (data/raw/2023/<species>/weekly/), then Matt flat layout
            try:
                stack, meta = load_weekly_stack(data_dir, species, resolution=args.resolution, year=args.year or 2023)
            except (FileNotFoundError, Exception):
                stack, meta = load_matt_stack(data_dir, species, resolution=args.resolution, year=args.year)
        except FileNotFoundError as e:
            print(f"  Skip: {e}")
            continue

        # Load date names and species season data
        species_json = {}
        try:
            with open(labels_path) as f:
                all_species_data = json.load(f)
            species_json = all_species_data.get(species, {})
            date_names = species_json.get("DATE_NAMES", DEFAULT_DATE_NAMES)
        except Exception:
            date_names = DEFAULT_DATE_NAMES

        change = compute_weekly_change(stack)

        if args.weekly:
            plot_weekly_movement_maps(
                output_dir=output_dir,
                species=species,
                stack=stack,
                date_names=date_names,
                meta=meta,
                change=change,
                use_basemap=args.basemap,
                region=args.region,
                weeks=args.weeks,
            )

        if args.onset:
            # Derive per-species search windows from eBird season dates + buffer,
            # falling back to fixed defaults if not available.
            windows = get_species_search_windows(species_json, date_names, buffer_weeks=args.season_buffer)
            cli_override = args.onset_spring_start is not None

            spring_start = args.onset_spring_start if args.onset_spring_start is not None else (windows["spring_start"] if windows else 5)
            spring_end = args.onset_spring_end if args.onset_spring_end is not None else (windows["spring_end"] if windows else 30)
            fall_start = args.onset_fall_start if args.onset_fall_start is not None else (windows["fall_start"] if windows else 30)
            fall_end = args.onset_fall_end if args.onset_fall_end is not None else (windows["fall_end"] if windows else 50)

            if cli_override:
                print(f"  Using CLI override search windows for {species}")
            elif windows:
                print(f"  Using eBird season dates ±{args.season_buffer}wk buffer for {species}")
            else:
                print(f"  Using fixed default search windows for {species} (not found in season JSON)")

            print(f"  Computing spring onset (weeks {spring_start}–{spring_end}, "
                  f"{date_names[spring_start]}–{date_names[min(spring_end - 1, len(date_names) - 1)]})...")
            onset_spring = compute_cell_onset(
                change, cell_size=args.cell_size,
                z_threshold=args.z_threshold,
                search_start=spring_start,
                search_end=spring_end,
            )
            plot_onset_map(
                output_dir=output_dir,
                species=species,
                onset=onset_spring,
                stack=stack,
                date_names=date_names,
                meta=meta,
                use_basemap=args.basemap,
                region=args.region,
                cell_size=args.cell_size,
                season="spring",
                search_start=spring_start,
                search_end=spring_end,
                display_buffer=args.display_weeks,
                cap_weeks=args.cap_weeks,
                clean=args.clean,
            )
            src = output_dir / f"{species}_migration_onset.png"
            dst = output_dir / f"{species}_spring_onset.png"
            src.rename(dst)
            print(f"  Saved as {dst}")

            print(f"  Computing fall onset (weeks {fall_start}–{fall_end}, "
                  f"{date_names[fall_start]}–{date_names[min(fall_end - 1, len(date_names) - 1)]})...")
            onset_fall = compute_cell_onset(
                change, cell_size=args.cell_size,
                z_threshold=args.z_threshold,
                search_start=fall_start,
                search_end=fall_end,
            )
            plot_onset_map(
                output_dir=output_dir,
                species=species,
                onset=onset_fall,
                stack=stack,
                date_names=date_names,
                meta=meta,
                use_basemap=args.basemap,
                region=args.region,
                cell_size=args.cell_size,
                season="fall",
                search_start=fall_start,
                search_end=fall_end,
                display_buffer=args.display_weeks,
                cap_weeks=args.cap_weeks,
                clean=args.clean,
            )
            src = output_dir / f"{species}_migration_onset.png"
            dst = output_dir / f"{species}_fall_onset.png"
            src.rename(dst)
            print(f"  Saved as {dst}")

    print(f"\nDone. Maps saved to {output_dir}")


if __name__ == "__main__":
    main()