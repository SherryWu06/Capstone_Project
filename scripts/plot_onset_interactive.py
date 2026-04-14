"""
Export interactive HTML onset maps: hover shows week index and calendar label.

Uses the same preprocessing as scripts/plot_migration_onset.py (prepare_onset_map_layers).

Run from project root:
    python scripts/plot_onset_interactive.py --species acafly --region north_america
    python scripts/plot_onset_interactive.py --species acafly --basemap

Outputs (per species):
    {species}_spring_onset_interactive.html
    {species}_fall_onset_interactive.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(_scripts_dir))

import plotly.graph_objects as go

import plot_migration_onset as pmo
from src.raster_processing import load_matt_stack, load_weekly_stack


def _week_date_hover_strings(onset_ll: np.ndarray, date_names: list[str]) -> np.ndarray:
    """2D array of hover text; vectorized except small Python string step on flat valid indices."""

    vn = np.isfinite(onset_ll)
    out = np.empty(onset_ll.shape, dtype=object)
    out[~vn] = "No onset (nodata)"
    if not np.any(vn):
        return out
    wi = np.round(onset_ll[vn]).astype(np.int32)
    n = len(date_names)
    lines = []
    for w in wi.flat:
        label = date_names[w] if 0 <= w < n else f"W{w}"
        lines.append(f"Week index {w} · {label}")
    out[vn] = lines
    return out



def _geo_border_traces(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> list:
    """
    Return Plotly Scatter traces for Natural Earth country and state/province borders,
    clipped to the given bounding box.  Uses the same cached GeoDataFrames as
    plot_migration_onset.add_basemap so borders are consistent across outputs.
    """
    try:
        countries, states = pmo._load_borders()
    except Exception as exc:
        print(f"  Warning: could not load borders for basemap: {exc}")
        return []

    traces = []
    for gdf, lw, color in [
        (countries, 0.8, "#555555"),
        (states,    0.4, "#aaaaaa"),
    ]:
        lons: list = []
        lats: list = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
            for part in parts:
                if part.geom_type == "Polygon":
                    xs, ys = part.exterior.xy
                    lons.extend(list(xs) + [None])
                    lats.extend(list(ys) + [None])
                elif part.geom_type == "LineString":
                    xs, ys = part.xy
                    lons.extend(list(xs) + [None])
                    lats.extend(list(ys) + [None])
        if lons:
            traces.append(
                go.Scatter(
                    x=lons,
                    y=lats,
                    mode="lines",
                    line=dict(width=lw, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def export_onset_plotly_html(
    *,
    layers: dict,
    species: str,
    season: str,
    date_names: list[str],
    output_path: Path,
    title_clean: bool = False,
    use_basemap: bool = False,
) -> None:
    onset_ll = layers["onset_ll"]
    lon_min = layers["lon_min"]
    lon_max = layers["lon_max"]
    lat_min = layers["lat_min"]
    lat_max = layers["lat_max"]
    week_min_display = layers["week_min_display"]
    week_max_display = layers["week_max_display"]
    date_start_label = layers["date_start_label"]
    date_end_label = layers["date_end_label"]

    h, w = onset_ll.shape
    x_coords = np.linspace(lon_min, lon_max, w)
    # Match matplotlib imshow(..., origin="upper"): row 0 = north (high latitude)
    y_coords = np.linspace(lat_max, lat_min, h)
    hover_m = _week_date_hover_strings(onset_ll, date_names)

    if title_clean:
        title = f"{species.upper()} · {season} onset"
    else:
        title = (
            f"{species.upper()} – Migration onset ({season})<br>"
            f"<sup>Blue = earliest, red = latest; {date_start_label}–{date_end_label}</sup>"
        )

    heatmap = go.Heatmap(
        x=x_coords,
        y=y_coords,
        z=onset_ll,
        zmin=float(week_min_display),
        zmax=float(week_max_display),
        colorscale="Turbo",
        colorbar=dict(title="Onset week index"),
        text=hover_m,
        hovertemplate="lon=%{x:.4f}<br>lat=%{y:.4f}<br>%{text}<extra></extra>",
        showscale=True,
    )

    border_traces = _geo_border_traces(lon_min, lon_max, lat_min, lat_max) if use_basemap else []
    fig = go.Figure(data=[heatmap] + border_traces)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Longitude",
            constrain="domain",
            range=[lon_min, lon_max],
        ),
        yaxis=dict(
            title="Latitude",
            scaleanchor="x",
            scaleratio=1,
            range=[lat_min, lat_max],
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="closest",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  Saved interactive map (Plotly) to {output_path}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive HTML onset maps (hover: week + date label)"
    )
    parser.add_argument("--species", nargs="+", default=["acafly", "comyel", "casvir"])
    parser.add_argument("--resolution", default="27km")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument(
        "--cell-size", type=int, default=16, help="Cell size for onset detection (default: 16)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/presentation",
        help="Output directory (default: outputs/presentation)",
    )
    parser.add_argument(
        "--region",
        choices=["full", "north_america", "americas", "lower_48", "lower_48_plus"],
        default="lower_48",
    )
    parser.add_argument("--z-threshold", type=float, default=1.5)
    parser.add_argument("--display-weeks", type=int, default=0)
    parser.add_argument("--cap-weeks", type=int, default=None)
    parser.add_argument("--clean", action="store_true", help="Shorter chart title")
    parser.add_argument(
        "--onset-spring-start", type=int, default=None,
    )
    parser.add_argument(
        "--onset-spring-end", type=int, default=None,
    )
    parser.add_argument(
        "--onset-fall-start", type=int, default=None,
    )
    parser.add_argument(
        "--onset-fall-end", type=int, default=None,
    )
    parser.add_argument(
        "--season-buffer", type=int, default=pmo.SEASON_BUFFER_WEEKS,
    )
    parser.add_argument(
        "--basemap",
        action="store_true",
        help="Overlay Natural Earth country and state/province borders on Plotly maps",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data" / "raw"
    labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"

    for species in args.species:
        print(f"\nProcessing {species} (interactive)...")
        try:
            try:
                stack, meta = load_weekly_stack(
                    data_dir, species, resolution=args.resolution, year=args.year or 2023
                )
            except (FileNotFoundError, Exception):
                stack, meta = load_matt_stack(
                    data_dir, species, resolution=args.resolution, year=args.year
                )
        except FileNotFoundError as e:
            print(f"  Skip: {e}")
            continue

        species_json: dict = {}
        try:
            with open(labels_path, encoding="utf-8") as f:
                all_species_data = json.load(f)
            species_json = all_species_data.get(species, {})
            date_names = species_json.get("DATE_NAMES", pmo.DEFAULT_DATE_NAMES)
        except OSError:
            date_names = pmo.DEFAULT_DATE_NAMES

        change = pmo.compute_weekly_change(stack)
        windows = pmo.get_species_search_windows(
            species_json, date_names, buffer_weeks=args.season_buffer
        )
        cli_override = args.onset_spring_start is not None

        spring_start = (
            args.onset_spring_start
            if args.onset_spring_start is not None
            else (windows["spring_start"] if windows else 5)
        )
        spring_end = (
            args.onset_spring_end
            if args.onset_spring_end is not None
            else (windows["spring_end"] if windows else 30)
        )
        fall_start = (
            args.onset_fall_start
            if args.onset_fall_start is not None
            else (windows["fall_start"] if windows else 30)
        )
        fall_end = (
            args.onset_fall_end
            if args.onset_fall_end is not None
            else (windows["fall_end"] if windows else 50)
        )

        if cli_override:
            print(f"  Using CLI override search windows for {species}")
        elif windows:
            print(f"  Using eBird season dates ±{args.season_buffer}wk buffer for {species}")
        else:
            print(f"  Using fixed default search windows for {species} (not found in season JSON)")

        print(
            f"  Spring onset weeks {spring_start}–{spring_end} "
            f"({date_names[spring_start]}–{date_names[min(spring_end - 1, len(date_names) - 1)]})..."
        )
        onset_spring = pmo.compute_cell_onset(
            change,
            cell_size=args.cell_size,
            z_threshold=args.z_threshold,
            search_start=spring_start,
            search_end=spring_end,
        )
        layers_sp = pmo.prepare_onset_map_layers(
            onset=onset_spring,
            stack=stack,
            date_names=date_names,
            meta=meta,
            region=args.region,
            cell_size=args.cell_size,
            search_start=spring_start,
            search_end=spring_end,
            display_buffer=args.display_weeks,
            cap_weeks=args.cap_weeks,
        )
        if layers_sp is None:
            print(f"  No spring onset for {species}, skipping spring HTML.")
        else:
            export_onset_plotly_html(
                layers=layers_sp,
                species=species,
                season="spring",
                date_names=date_names,
                output_path=output_dir / f"{species}_spring_onset_interactive.html",
                title_clean=args.clean,
                use_basemap=args.basemap,
            )

        print(
            f"  Fall onset weeks {fall_start}–{fall_end} "
            f"({date_names[fall_start]}–{date_names[min(fall_end - 1, len(date_names) - 1)]})..."
        )
        onset_fall = pmo.compute_cell_onset(
            change,
            cell_size=args.cell_size,
            z_threshold=args.z_threshold,
            search_start=fall_start,
            search_end=fall_end,
        )
        layers_fa = pmo.prepare_onset_map_layers(
            onset=onset_fall,
            stack=stack,
            date_names=date_names,
            meta=meta,
            region=args.region,
            cell_size=args.cell_size,
            search_start=fall_start,
            search_end=fall_end,
            display_buffer=args.display_weeks,
            cap_weeks=args.cap_weeks,
        )
        if layers_fa is None:
            print(f"  No fall onset for {species}, skipping fall HTML.")
        else:
            export_onset_plotly_html(
                layers=layers_fa,
                species=species,
                season="fall",
                date_names=date_names,
                output_path=output_dir / f"{species}_fall_onset_interactive.html",
                title_clean=args.clean,
                use_basemap=args.basemap,
            )

    print(f"\nDone. Interactive HTML saved under {output_dir}")


if __name__ == "__main__":
    main()
