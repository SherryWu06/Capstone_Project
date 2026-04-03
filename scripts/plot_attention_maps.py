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

from src.raster_processing import (
    load_matt_stack,
    load_weekly_stack,
    find_species_data,
    load_config,
    list_ebirdst_species,
)


def load_abundance_stack(
    data_dir: Path,
    species: str,
    resolution: str = "27km",
    year: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Matt flat layout first; then ebirdst data/raw/{year}/{species}/."""
    try:
        return load_matt_stack(data_dir, species, resolution=resolution, year=year)
    except FileNotFoundError:
        y = year if year is not None else 2023
        return load_weekly_stack(data_dir, species, resolution=resolution, year=y)


def get_date_names_for_species(
    data_dir: Path,
    species: str,
    labels_path: Path,
    year: int | None = None,
) -> list[str]:
    """Matt JSON DATE_NAMES, or ebirdst species config.json."""
    if labels_path.exists():
        try:
            with open(labels_path) as f:
                data = json.load(f)
            dn = data.get(species, {}).get("DATE_NAMES", [])
            if dn:
                return dn
        except Exception:
            pass
    try:
        y = year if year is not None else 2023
        species_dir = find_species_data(data_dir, species, y)
        cfg = load_config(species_dir)
        return cfg.get("DATE_NAMES", [])
    except Exception:
        return []

# Region bounds (lon_min, lon_max, lat_min, lat_max) in WGS84
REGION_BOUNDS = {
    "north_america": (-170, -50, 15, 72),
    "americas": (-170, -35, -55, 72),
    # Lower 48 + buffer into southern Canada/Alaska panhandle and northern Mexico
    "lower_48_wide": (-130, -60, 18, 58),
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


def get_extent_for_region(region: str, crs, full_extent: list) -> tuple:
    """
    Returns (plot_extent, use_geo) where:
      - plot_extent is the extent to pass to ax.set_extent
      - use_geo is True if the extent is in WGS84 (lon/lat) coordinates,
        False if it is in the data's projected CRS coordinates.
    Using WGS84 extents with ccrs.PlateCarree() avoids the slanted/skewed
    view that occurs when projected bounds are passed directly.
    """
    if region == "full" or region not in REGION_BOUNDS:
        return full_extent, False
    lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
    return [lon_min, lon_max, lat_min, lat_max], True


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
    # Load raster (stack + metadata): Matt or ebirdst
    try:
        stack, meta = load_abundance_stack(data_dir, species, resolution=resolution, year=year)
    except FileNotFoundError:
        print(f"  Skip {species}: raster not found (Matt or ebirdst)")
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

    plot_extent, extent_is_geo = get_extent_for_region(region, crs, extent)

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
                extent_crs = ccrs.PlateCarree() if extent_is_geo else proj
                ax.set_extent(plot_extent, crs=extent_crs)
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
        stack, meta = load_abundance_stack(data_dir, species, resolution=resolution, year=year)
    except FileNotFoundError:
        print(f"  Skip {species}: raster not found (Matt or ebirdst)")
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

    # Load DATE_NAMES for week labels (Matt JSON or ebirdst config)
    date_names = get_date_names_for_species(data_dir, species, labels_path, year=year)
    if not date_names:
        date_names = [f"W{i:02d}" for i in range(len(week_files))]

    bounds = array_bounds(h, w, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

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

    plot_extent, extent_is_geo = get_extent_for_region(region, crs, extent)
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

    # Parse filenames and load+upsample arrays once; cache for the render loop.
    ones_tile = np.ones((cell_size, cell_size))
    parsed: list[tuple[int, str, np.ndarray]] = []
    class_vmax: dict[str, float] = {}
    for week_path in week_files:
        parts = week_path.stem.replace("attention_week", "").split("_")
        week_idx = int(parts[0]) if parts else 0
        label_name = parts[1] if len(parts) > 1 else "unknown"
        attn_up = np.kron(np.load(week_path), ones_tile)[:h, :w]
        p99 = float(np.percentile(attn_up, 99))
        class_vmax[label_name] = max(class_vmax.get(label_name, 0.0), p99)
        parsed.append((week_idx, label_name, attn_up))

    for week_idx, label_name, attn_upsampled in parsed:
        date_str = date_names[week_idx] if week_idx < len(date_names) else f"Week {week_idx}"
        title_label = LABEL_MAP.get(label_name, label_name.replace("_", " ").title())

        # Use the shared vmax for this label class (falls back to per-map if missing)
        vmax = class_vmax.get(label_name, float(np.percentile(attn_upsampled, 99)))

        subplot_kw = {"projection": proj} if (proj is not None) else {}
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw=subplot_kw)

        if proj is not None:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="gray")
            try:
                extent_crs = ccrs.PlateCarree() if extent_is_geo else proj
                ax.set_extent(plot_extent, crs=extent_crs)
            except (ValueError, TypeError):
                pass

        if overlay_abundance and ab_norm is not None:
            ax.imshow(ab_norm, cmap="Greys", alpha=0.8, **imshow_kw)
        im = ax.imshow(
            attn_upsampled,
            cmap="viridis",
            alpha=0.7,
            norm=mcolors.Normalize(vmin=0, vmax=vmax),
            **imshow_kw,
        )
        ax.set_title(f"{display_name} – Week of {date_str} ({title_label})", fontsize=14)
        plt.colorbar(im, ax=ax, label="Model importance (shared scale per label)", shrink=0.8)
        plt.tight_layout()

        out_path = weekly_dir / f"{species}_week{week_idx:02d}_{date_str.replace('-', '')}_{label_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(parsed)} weekly maps to {weekly_dir}")


def plot_attention_difference_map(
    output_dir: Path,
    attention_dir: Path,
    species: str,
    data_dir: Path,
    cell_size: int = 16,
    resolution: str = "27km",
    year: int = None,
    overlay_abundance: bool = True,
    use_basemap: bool = False,
    region: str = "full",
    movement_key: str = "movement",
    no_movement_key: str = "no_movement",
) -> None:
    """
    Plot signed attention difference (migration − non-migration) on a diverging colormap.

    Red regions = model attends more during migration weeks.
    Blue regions = model attends more during breeding/wintering weeks.
    Requires attention_aggregate_{movement_key}.npy and attention_aggregate_{no_movement_key}.npy.
    """
    sp_dir = attention_dir / species
    movement_path = sp_dir / f"attention_aggregate_{movement_key}.npy"
    no_movement_path = sp_dir / f"attention_aggregate_{no_movement_key}.npy"

    if not movement_path.exists() or not no_movement_path.exists():
        print(
            f"  Skip {species} difference map: missing aggregate file(s) "
            f"({movement_path.name} or {no_movement_path.name})"
        )
        return

    try:
        stack, meta = load_abundance_stack(data_dir, species, resolution=resolution, year=year)
    except FileNotFoundError:
        print(f"  Skip {species} difference map: raster not found")
        return

    transform = meta["transform"]
    crs = meta["crs"]
    h, w = meta["height"], meta["width"]

    stack_float = stack.astype(np.float64)
    stack_float[~np.isfinite(stack_float)] = np.nan
    mean_abundance = np.nanmean(stack_float, axis=0)

    attn_movement = np.load(movement_path)
    attn_no_movement = np.load(no_movement_path)

    # Upsample both aggregate maps to raster resolution
    attn_movement_up = np.kron(attn_movement, np.ones((cell_size, cell_size)))[:h, :w]
    attn_no_movement_up = np.kron(attn_no_movement, np.ones((cell_size, cell_size)))[:h, :w]

    # Signed difference: positive = model focuses here more during migration
    diff = attn_movement_up - attn_no_movement_up
    abs_max = np.percentile(np.abs(diff), 99)

    bounds = array_bounds(h, w, transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

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

    plot_extent, extent_is_geo = get_extent_for_region(region, crs, extent)

    imshow_kw = {"extent": extent, "origin": "upper", "aspect": "auto"}
    if proj is not None:
        imshow_kw["transform"] = proj

    subplot_kw = {"projection": proj} if proj is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), subplot_kw=subplot_kw)

    if proj is not None:
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="gray")
        try:
            extent_crs = ccrs.PlateCarree() if extent_is_geo else proj
            ax.set_extent(plot_extent, crs=extent_crs)
        except (ValueError, TypeError):
            pass

    if overlay_abundance:
        ab_norm = np.nan_to_num(mean_abundance, nan=0)
        p95 = np.percentile(ab_norm[ab_norm > 0], 95) + 1e-9
        ab_norm = np.clip(ab_norm / p95, 0, 1)
        ax.imshow(ab_norm, cmap="Greys", alpha=0.6, **imshow_kw)

    im = ax.imshow(
        diff,
        cmap="RdBu_r",
        alpha=0.8,
        norm=mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max),
        **imshow_kw,
    )

    display_name = get_display_name(species)
    ax.set_title(
        f"{display_name} – Annual Attention Contrast\n"
        "Red = migration focus  |  Blue = breeding/wintering focus",
        fontsize=13,
    )
    ax.set_xlabel("Easting (m)" if proj is None else "Longitude")
    ax.set_ylabel("Northing (m)" if proj is None else "Latitude")

    cbar = plt.colorbar(im, ax=ax, label="Attention: Migration − Breeding & Wintering", shrink=0.8)
    cbar.ax.axhline(y=0.5, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    out_path = output_dir / f"{species}_attention_difference.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


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
        help="Species to plot (default: acafly comyel casvir). Ignored if --ebirdst-all is set.",
    )
    parser.add_argument(
        "--ebirdst-all",
        action="store_true",
        help=(
            "Discover species from data/raw/{year}/ that have abundance_median weekly TIFs at "
            "--resolution (same rule as run_baseline.py --ebirdst-all). Overrides --species."
        ),
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
        choices=["full", "north_america", "americas", "lower_48_wide"],
        default="lower_48_wide",
        help="Zoom extent: full, north_america, americas, or lower_48_wide (default: lower_48_wide)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help=(
            "Generate a signed attention-difference map per species "
            "(migration − breeding/wintering) on a red/blue diverging colormap"
        ),
    )
    args = parser.parse_args()

    if args.basemap and not CARTOPY_AVAILABLE:
        print("Warning: cartopy not installed. Run: pip install cartopy")
        print("  Falling back to no basemap.")
        args.basemap = False

    if args.region != "full" and not CARTOPY_AVAILABLE:
        print("Warning: cartopy not installed; --region zoom requires cartopy + --basemap.")
        args.region = "full"

    attention_dir = Path(args.attention_dir)
    output_dir = Path(args.output_dir)
    data_dir = project_root / "data" / "raw"
    labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ebirdst_all:
        plot_year = args.year if args.year is not None else 2023
        args.species = list_ebirdst_species(
            data_dir, resolution=args.resolution, year=plot_year
        )
        if not args.species:
            print(
                "Error: --ebirdst-all found no species with median rasters at "
                f"{args.resolution} under {data_dir}/{plot_year}/.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  --ebirdst-all: {len(args.species)} species at {args.resolution} ({plot_year})")

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
        if args.diff:
            plot_attention_difference_map(
                output_dir=output_dir,
                attention_dir=attention_dir,
                species=species,
                data_dir=data_dir,
                cell_size=args.cell_size,
                resolution=args.resolution,
                year=args.year,
                overlay_abundance=not args.no_overlay,
                use_basemap=args.basemap,
                region=args.region,
            )

    print(f"Done. Maps saved to {output_dir}")


if __name__ == "__main__":
    main()
