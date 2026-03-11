"""
Create presentation-quality maps of MIL attention for a bird-focused audience.

Overlays attention on mean abundance. Georeferenced with clear labels.
Run from project root: python scripts/plot_attention_maps.py [options]
Requires outputs from: run_baseline.py --matt --species-split --regional --output-dir

To see maps on a US base: use --basemap (adds coastlines/states) or --geotiff (for QGIS).
Use --weekly for per-week maps; --region to zoom to North America or Americas.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import Affine, array_bounds

# Optional: cartopy for basemap (coastlines, US states)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Optional: pyproj for region bounds
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.raster_processing import load_matt_stack

# Region bounds (lon_min, lon_max, lat_min, lat_max) in WGS84
REGION_BOUNDS = {
    "north_america": (-170, -50, 15, 72),
    "americas": (-170, -35, -55, 72),
}

# Bird-friendly labels (technical -> presentation)
LABEL_MAP = {
    "movement": "Migration",
    "no_movement": "Breeding & Wintering",
    "breeding": "Breeding",
    "nonbreeding": "Wintering",
    "prebreeding_migration": "Spring Migration",
    "postbreeding_migration": "Fall Migration",
}


def get_display_name(species_code: str) -> str:
    """Convert species code to title case for display (e.g., acafly -> Acafly)."""
    return species_code.replace("_", " ").title()


def get_extent_for_region(region: str, crs, full_extent: list) -> list:
    """
    Return extent [west, east, south, north] for the given region.
    full_extent: bounds from raster in raster CRS.
    """
    if region == "full" or region not in REGION_BOUNDS:
        return full_extent
    if not PYPROJ_AVAILABLE:
        return full_extent
    try:
        lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        trans = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        west, south = trans.transform(lon_min, lat_min)
        east, north = trans.transform(lon_max, lat_max)
        # Clip to raster bounds
        w0, e0, s0, n0 = full_extent[0], full_extent[1], full_extent[2], full_extent[3]
        return [
            max(west, w0), min(east, e0),
            max(south, s0), min(north, n0),
        ]
    except Exception:
        return full_extent


def plot_attention_maps(
    output_dir: Path,
    attention_dir: Path,
    species: str,
    data_dir: Path,
    cell_size: int = 16,
    resolution: str = "27km",
    year: int = None,
    save_geotiff: bool = False,
    overlay_abundance: bool = True,
    use_basemap: bool = False,
    region: str = "full",
) -> None:
    """
    Create georeferenced attention maps for one species.
    Overlays attention on mean abundance when overlay_abundance=True.
    """
    # Load raster (stack + metadata)
    try:
        stack, meta = load_matt_stack(data_dir, species, resolution=resolution, year=year)
    except FileNotFoundError:
        print(f"  Skip {species}: raster not found")
        return

    transform = meta["transform"]
    crs = meta["crs"]
    h, w = meta["height"], meta["width"]

    # Mean abundance over weeks (background)
    stack_float = stack.astype(np.float64)
    stack_float[~np.isfinite(stack_float)] = np.nan
    mean_abundance = np.nanmean(stack_float, axis=0)

    # Load aggregate attention maps
    sp_dir = attention_dir / species
    if not sp_dir.exists():
        print(f"  Skip {species}: no attention data at {sp_dir}")
        return

    aggregate_files = list(sp_dir.glob("attention_aggregate_*.npy"))
    if not aggregate_files:
        print(f"  Skip {species}: no aggregate files")
        return

    display_name = get_display_name(species)
    bounds = array_bounds(h, w, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    plot_extent = get_extent_for_region(region, crs, extent)

    # Projection for basemap (match raster CRS)
    proj = None
    if use_basemap and CARTOPY_AVAILABLE:
        try:
            epsg = int(str(crs).replace("EPSG:", "")) if crs else 5070
            proj = ccrs.epsg(epsg)
        except Exception:
            proj = ccrs.AlbersEqualArea(
                central_longitude=-96, central_latitude=23,
                standard_parallels=(29.5, 45.5),
            )

    n_maps = len(aggregate_files)
    subplot_kw = {"projection": proj} if (proj is not None) else {}
    fig, axes = plt.subplots(1, n_maps, figsize=(6 * n_maps, 5), subplot_kw=subplot_kw)
    if n_maps == 1:
        axes = [axes]

    imshow_kw = {"extent": extent, "origin": "upper", "aspect": "auto"}
    if proj is not None:
        imshow_kw["transform"] = proj

    for ax, agg_path in zip(axes, sorted(aggregate_files)):
        label_key = agg_path.stem.replace("attention_aggregate_", "")
        title = LABEL_MAP.get(label_key, label_key.replace("_", " ").title())

        if proj is not None:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="gray")
            try:
                ax.set_extent(plot_extent, crs=proj)
            except (ValueError, TypeError):
                pass  # fallback: imshow extent will define view

        attn = np.load(agg_path)
        n_rows, n_cols = attn.shape

        # Upsample attention to raster resolution (repeat each cell)
        attn_upsampled = np.kron(attn, np.ones((cell_size, cell_size)))
        attn_upsampled = attn_upsampled[:h, :w]

        if overlay_abundance:
            # Background: mean abundance (grayscale, darker = more birds)
            ab_norm = np.nan_to_num(mean_abundance, nan=0)
            ab_norm = np.clip(ab_norm / (np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9), 0, 1)
            ax.imshow(ab_norm, cmap="Greys", alpha=0.8, **imshow_kw)
            # Overlay: attention (semi-transparent viridis)
            im = ax.imshow(
                attn_upsampled,
                cmap="viridis",
                alpha=0.7,
                norm=mcolors.Normalize(vmin=0, vmax=np.percentile(attn_upsampled, 99)),
                **imshow_kw,
            )
        else:
            im = ax.imshow(attn_upsampled, cmap="viridis", **imshow_kw)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Easting (m)" if proj is None else "Longitude")
        ax.set_ylabel("Northing (m)" if proj is None else "Latitude")
        plt.colorbar(im, ax=ax, label="Model importance", shrink=0.8)

    fig.suptitle(
        f"{display_name} ({species}) – Model focus overlaid on abundance",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()

    out_path = output_dir / f"{species}_attention_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    if save_geotiff:
        cell_transform = transform * Affine.scale(cell_size, cell_size)
        for agg_path in sorted(aggregate_files):
            label_key = agg_path.stem.replace("attention_aggregate_", "")
            attn = np.load(agg_path)
            attn_up = np.kron(attn, np.ones((cell_size, cell_size)))[:h, :w]
            tif_path = output_dir / f"{species}_attention_{label_key}.tif"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=attn_up.shape[0],
                width=attn_up.shape[1],
                count=1,
                dtype=attn_up.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(attn_up, 1)
            print(f"  Saved {tif_path}")


def plot_weekly_attention_maps(
    output_dir: Path,
    attention_dir: Path,
    species: str,
    data_dir: Path,
    labels_path: Path,
    cell_size: int = 16,
    resolution: str = "27km",
    year: int = None,
    overlay_abundance: bool = True,
    use_basemap: bool = False,
    region: str = "full",
    weeks: list[int] | None = None,
) -> None:
    """
    Create one map per week for a species. Saves to output_dir/weekly/{species}/.
    """
    try:
        stack, meta = load_matt_stack(data_dir, species, resolution=resolution, year=year)
    except FileNotFoundError:
        print(f"  Skip {species}: raster not found")
        return

    transform = meta["transform"]
    crs = meta["crs"]
    h, w = meta["height"], meta["width"]

    stack_float = stack.astype(np.float64)
    stack_float[~np.isfinite(stack_float)] = np.nan
    mean_abundance = np.nanmean(stack_float, axis=0)

    sp_dir = attention_dir / species
    if not sp_dir.exists():
        print(f"  Skip {species}: no attention data at {sp_dir}")
        return

    week_files = sorted(sp_dir.glob("attention_week*.npy"))
    if not week_files:
        print(f"  Skip {species}: no weekly attention files")
        return

    if weeks is not None:
        week_set = set(weeks)
        week_files = [f for f in week_files if int(f.stem.replace("attention_week", "").split("_")[0]) in week_set]
        if not week_files:
            print(f"  Skip {species}: no matching weeks in {weeks}")
            return

    # Load DATE_NAMES for week labels
    try:
        with open(labels_path) as f:
            data = json.load(f)
        date_names = data.get(species, {}).get("DATE_NAMES", [])
    except Exception:
        date_names = [f"W{i:02d}" for i in range(len(week_files))]

    bounds = array_bounds(h, w, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    plot_extent = get_extent_for_region(region, crs, extent)

    proj = None
    if use_basemap and CARTOPY_AVAILABLE:
        try:
            epsg = int(str(crs).replace("EPSG:", "")) if crs else 5070
            proj = ccrs.epsg(epsg)
        except Exception:
            proj = ccrs.AlbersEqualArea(
                central_longitude=-96, central_latitude=23,
                standard_parallels=(29.5, 45.5),
            )

    weekly_dir = output_dir / "weekly" / species
    weekly_dir.mkdir(parents=True, exist_ok=True)

    imshow_kw = {"extent": extent, "origin": "upper", "aspect": "auto"}
    if proj is not None:
        imshow_kw["transform"] = proj

    display_name = get_display_name(species)
    ab_norm = None
    if overlay_abundance:
        ab_norm = np.nan_to_num(mean_abundance, nan=0)
        p95 = np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9
        ab_norm = np.clip(ab_norm / p95, 0, 1)

    for week_path in week_files:
        # Parse week index from filename: attention_week21_movement.npy -> 21
        stem = week_path.stem
        parts = stem.replace("attention_week", "").split("_")
        week_idx = int(parts[0]) if parts else 0
        label_name = parts[1] if len(parts) > 1 else "unknown"

        date_str = date_names[week_idx] if week_idx < len(date_names) else f"Week {week_idx}"
        title_label = LABEL_MAP.get(label_name, label_name.replace("_", " ").title())

        attn = np.load(week_path)
        attn_upsampled = np.kron(attn, np.ones((cell_size, cell_size)))
        attn_upsampled = attn_upsampled[:h, :w]

        subplot_kw = {"projection": proj} if (proj is not None) else {}
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw=subplot_kw)

        if proj is not None:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="gray")
            try:
                ax.set_extent(plot_extent, crs=proj)
            except (ValueError, TypeError):
                pass

        if overlay_abundance and ab_norm is not None:
            ax.imshow(ab_norm, cmap="Greys", alpha=0.8, **imshow_kw)
        im = ax.imshow(
            attn_upsampled,
            cmap="viridis",
            alpha=0.7,
            norm=mcolors.Normalize(vmin=0, vmax=np.percentile(attn_upsampled, 99)),
            **imshow_kw,
        )
        ax.set_title(f"{display_name} – Week of {date_str} ({title_label})", fontsize=14)
        plt.colorbar(im, ax=ax, label="Model importance", shrink=0.8)
        plt.tight_layout()

        out_path = weekly_dir / f"{species}_week{week_idx:02d}_{date_str.replace('-', '')}_{label_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(week_files)} weekly maps to {weekly_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create presentation maps of MIL attention")
    parser.add_argument(
        "--attention-dir",
        type=str,
        default="outputs/mil_inspect/test",
        help="Directory with attention outputs (default: outputs/mil_inspect/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/presentation",
        help="Where to save maps (default: outputs/presentation)",
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=["acafly", "comyel", "casvir"],
        help="Species to plot (default: acafly comyel casvir)",
    )
    parser.add_argument("--cell-size", type=int, default=16, help="Cell size used in MIL (default: 16)")
    parser.add_argument("--resolution", type=str, default="27km")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--geotiff", action="store_true", help="Also save GeoTIFFs for QGIS")
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not overlay on abundance (show attention only)",
    )
    parser.add_argument(
        "--basemap",
        action="store_true",
        help="Add coastlines and US state borders (requires cartopy)",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Generate per-week attention maps (saved to output_dir/weekly/{species}/)",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="*",
        default=None,
        help="Limit weekly maps to these week indices (e.g. --weeks 20 21 22 23 24 for breeding transition)",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["full", "north_america", "americas"],
        default="full",
        help="Zoom extent: full, north_america, or americas (default: full)",
    )
    args = parser.parse_args()

    if args.basemap and not CARTOPY_AVAILABLE:
        print("Warning: cartopy not installed. Run: pip install cartopy")
        print("  Falling back to no basemap.")
        args.basemap = False

    if args.region != "full" and not PYPROJ_AVAILABLE:
        print("Warning: pyproj not installed. Run: pip install pyproj for --region zoom")
        print("  Using full extent.")
        args.region = "full"

    attention_dir = Path(args.attention_dir)
    output_dir = Path(args.output_dir)
    data_dir = project_root / "data" / "raw"
    labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating presentation maps...")
    for species in args.species:
        plot_attention_maps(
            output_dir=output_dir,
            attention_dir=attention_dir,
            species=species,
            data_dir=data_dir,
            cell_size=args.cell_size,
            resolution=args.resolution,
            year=args.year,
            save_geotiff=args.geotiff,
            overlay_abundance=not args.no_overlay,
            use_basemap=args.basemap,
            region=args.region,
        )
        if args.weekly:
            plot_weekly_attention_maps(
                output_dir=output_dir,
                attention_dir=attention_dir,
                species=species,
                data_dir=data_dir,
                labels_path=labels_path,
                cell_size=args.cell_size,
                resolution=args.resolution,
                year=args.year,
                overlay_abundance=not args.no_overlay,
                use_basemap=args.basemap,
                region=args.region,
                weeks=args.weeks,
            )

    print(f"Done. Maps saved to {output_dir}")


if __name__ == "__main__":
    main()
