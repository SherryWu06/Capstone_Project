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
import matplotlib.cm as cm
import rasterio
from rasterio.transform import array_bounds

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.raster_processing import load_matt_stack

# Region bounds (lon_min, lon_max, lat_min, lat_max) in WGS84
REGION_BOUNDS = {
    "north_america": (-170, -50, 15, 72),
    "americas": (-170, -35, -55, 72),
}

DEFAULT_DATE_NAMES = [
    "01-04", "01-11", "01-18", "01-25", "02-01", "02-08", "02-15", "02-22",
    "03-01", "03-08", "03-15", "03-22", "03-29", "04-05", "04-12", "04-19", "04-26",
    "05-03", "05-10", "05-17", "05-24", "05-31", "06-07", "06-14", "06-21", "06-28",
    "07-05", "07-12", "07-19", "07-26", "08-02", "08-09", "08-16", "08-23", "08-30",
    "09-06", "09-13", "09-20", "09-27", "10-04", "10-11", "10-18", "10-25", "11-01",
    "11-08", "11-15", "11-22", "11-29", "12-06", "12-13", "12-20", "12-27",
]


def get_extent_for_region(region: str, crs, full_extent: list) -> list:
    if region == "full" or region not in REGION_BOUNDS:
        return full_extent
    if not PYPROJ_AVAILABLE:
        return full_extent
    try:
        lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        trans = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        west, south = trans.transform(lon_min, lat_min)
        east, north = trans.transform(lon_max, lat_max)
        w0, e0, s0, n0 = full_extent
        return [max(west, w0), min(east, e0), max(south, s0), min(north, n0)]
    except Exception:
        return full_extent


def get_cartopy_proj(crs):
    if not CARTOPY_AVAILABLE:
        return None
    try:
        epsg = int(str(crs).replace("EPSG:", ""))
        return ccrs.epsg(epsg)
    except Exception:
        return ccrs.AlbersEqualArea(
            central_longitude=-96, central_latitude=23,
            standard_parallels=(29.5, 45.5),
        )


def add_basemap_features(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)


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
    n_rows = max(1, h // cell_size)
    n_cols = max(1, w // cell_size)
    onset = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    for ri in range(n_rows):
        for ci in range(n_cols):
            r0, r1 = ri * cell_size, min((ri + 1) * cell_size, h)
            c0, c1 = ci * cell_size, min((ci + 1) * cell_size, w)

            # Mean change per week for this cell
            cell_chg = np.array([
                np.nanmean(change[t, r0:r1, c0:c1]) for t in range(n_weeks)
            ])

            mu = np.nanmean(cell_chg)
            sigma = np.nanstd(cell_chg)
            if sigma < 1e-9:
                continue  # no variation (ocean cell or constant abundance)

            z = (cell_chg - mu) / sigma
            for t in range(search_start, search_end):
                if z[t] >= z_threshold:
                    onset[ri, ci] = t
                    break

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
    h, w = meta["height"], meta["width"]
    crs = meta["crs"]
    transform = meta["transform"]
    bounds = array_bounds(h, w, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    plot_extent = get_extent_for_region(region, crs, extent)

    proj = get_cartopy_proj(crs) if use_basemap else None
    weekly_dir = output_dir / "movement" / species
    weekly_dir.mkdir(parents=True, exist_ok=True)

    # Compute mean abundance for background
    s = stack.astype(np.float64)
    s[~np.isfinite(s)] = np.nan
    mean_ab = np.nanmean(s, axis=0)
    ab_norm = np.nan_to_num(mean_ab, nan=0)
    p95 = np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9
    ab_norm = np.clip(ab_norm / p95, 0, 1)

    # Color scale: 99th percentile of all change values
    all_chg = change[1:]  # skip week 0
    vmax = float(np.nanpercentile(all_chg[all_chg > 0], 99)) if np.any(all_chg > 0) else 1.0

    imshow_kw = {"extent": extent, "origin": "upper", "aspect": "auto"}
    if proj is not None:
        imshow_kw["transform"] = proj

    n_weeks = change.shape[0]
    week_iter = weeks if weeks is not None else range(1, n_weeks)

    for t in week_iter:
        if t >= n_weeks:
            continue
        date_str = date_names[t] if t < len(date_names) else f"W{t:02d}"
        chg_t = change[t].copy()

        subplot_kw = {"projection": proj} if proj is not None else {}
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw=subplot_kw)

        if proj is not None:
            add_basemap_features(ax)
            try:
                ax.set_extent(plot_extent, crs=proj)
            except (ValueError, TypeError):
                pass

        ax.imshow(ab_norm, cmap="Greys", alpha=0.6, **imshow_kw)
        im = ax.imshow(
            chg_t,
            cmap="hot_r",
            alpha=0.8,
            norm=mcolors.Normalize(vmin=0, vmax=vmax),
            **imshow_kw,
        )
        ax.set_title(f"{species.upper()} – Movement week of {date_str}", fontsize=14)
        plt.colorbar(im, ax=ax, label="Abundance change (week-to-week)", shrink=0.8)
        plt.tight_layout()

        out_path = weekly_dir / f"{species}_movement_week{t:02d}_{date_str.replace('-', '')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved movement maps for {len(week_iter) if isinstance(week_iter, list) else n_weeks - 1} weeks to {weekly_dir}")


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
) -> None:
    """
    Plot onset week per cell as a spatial map.
    Each cell is colored by the week number when it first showed movement.
    """
    h, w = meta["height"], meta["width"]
    crs = meta["crs"]
    transform = meta["transform"]
    bounds = array_bounds(h, w, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    plot_extent = get_extent_for_region(region, crs, extent)

    proj = get_cartopy_proj(crs) if use_basemap else None

    # Upsample onset to raster resolution
    onset_up = np.kron(onset, np.ones((cell_size, cell_size)))
    onset_up = onset_up[:h, :w]
    onset_up[onset_up == 0] = np.nan  # 0 means no onset found

    # Background abundance
    s = stack.astype(np.float64)
    s[~np.isfinite(s)] = np.nan
    mean_ab = np.nanmean(s, axis=0)
    ab_norm = np.nan_to_num(mean_ab, nan=0)
    p95 = np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9
    ab_norm = np.clip(ab_norm / p95, 0, 1)

    valid_weeks = onset[np.isfinite(onset)]
    if len(valid_weeks) == 0:
        print(f"  No onset found for {species}, skipping onset map.")
        return

    week_min, week_max = int(valid_weeks.min()), int(valid_weeks.max())

    imshow_kw = {"extent": extent, "origin": "upper", "aspect": "auto"}
    if proj is not None:
        imshow_kw["transform"] = proj

    subplot_kw = {"projection": proj} if proj is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), subplot_kw=subplot_kw)

    if proj is not None:
        add_basemap_features(ax)
        try:
            ax.set_extent(plot_extent, crs=proj)
        except (ValueError, TypeError):
            pass

    ax.imshow(ab_norm, cmap="Greys", alpha=0.6, **imshow_kw)
    im = ax.imshow(
        onset_up,
        cmap="RdYlGn_r",  # red = early, green = late
        alpha=0.85,
        norm=mcolors.Normalize(vmin=week_min, vmax=week_max),
        **imshow_kw,
    )

    # Colorbar with date labels
    cbar = plt.colorbar(im, ax=ax, label="Migration onset (week)", shrink=0.7)
    tick_weeks = np.linspace(week_min, week_max, min(6, week_max - week_min + 1), dtype=int)
    cbar.set_ticks(tick_weeks)
    cbar.set_ticklabels([
        date_names[w] if w < len(date_names) else f"W{w}" for w in tick_weeks
    ])

    ax.set_title(
        f"{species.upper()} – Migration onset date by region\n"
        f"(Red = earliest, Green = latest; range {date_names[week_min]}–{date_names[min(week_max, len(date_names)-1)]})",
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
        choices=["full", "north_america", "americas"],
        default="full",
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
        "--onset-spring-start", type=int, default=5,
        help="First week to consider for spring onset search (default: 5 = ~Feb 1)",
    )
    parser.add_argument(
        "--onset-spring-end", type=int, default=30,
        help="Last week to consider for spring onset search (default: 30 = ~Aug 2)",
    )
    parser.add_argument(
        "--onset-fall-start", type=int, default=30,
        help="First week to consider for fall onset search (default: 30 = ~Aug 2)",
    )
    parser.add_argument(
        "--onset-fall-end", type=int, default=50,
        help="Last week to consider for fall onset search (default: 50 = ~Dec 13)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=1.5,
        help="Z-score threshold for onset detection (default: 1.5)",
    )
    args = parser.parse_args()

    if not args.weekly and not args.onset:
        print("No output mode selected. Use --weekly and/or --onset.")
        parser.print_help()
        return

    if args.basemap and not CARTOPY_AVAILABLE:
        print("Warning: cartopy not installed. Run: pip install cartopy")
        args.basemap = False
    if args.region != "full" and not PYPROJ_AVAILABLE:
        print("Warning: pyproj not installed. Run: pip install pyproj")
        args.region = "full"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data" / "raw"
    labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"

    for species in args.species:
        print(f"\nProcessing {species}...")
        try:
            stack, meta = load_matt_stack(data_dir, species, resolution=args.resolution, year=args.year)
        except FileNotFoundError as e:
            print(f"  Skip: {e}")
            continue

        # Load date names
        try:
            with open(labels_path) as f:
                data = json.load(f)
            date_names = data.get(species, {}).get("DATE_NAMES", DEFAULT_DATE_NAMES)
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
            print(f"  Computing spring onset (weeks {args.onset_spring_start}–{args.onset_spring_end})...")
            onset_spring = compute_cell_onset(
                change, cell_size=args.cell_size,
                z_threshold=args.z_threshold,
                search_start=args.onset_spring_start,
                search_end=args.onset_spring_end,
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
            )
            # Rename to indicate spring
            src = output_dir / f"{species}_migration_onset.png"
            dst = output_dir / f"{species}_spring_onset.png"
            src.rename(dst)
            print(f"  Saved as {dst}")

            print(f"  Computing fall onset (weeks {args.onset_fall_start}–{args.onset_fall_end})...")
            onset_fall = compute_cell_onset(
                change, cell_size=args.cell_size,
                z_threshold=args.z_threshold,
                search_start=args.onset_fall_start,
                search_end=args.onset_fall_end,
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
            )
            src = output_dir / f"{species}_migration_onset.png"
            dst = output_dir / f"{species}_fall_onset.png"
            src.rename(dst)
            print(f"  Saved as {dst}")

    print(f"\nDone. Maps saved to {output_dir}")


if __name__ == "__main__":
    main()
