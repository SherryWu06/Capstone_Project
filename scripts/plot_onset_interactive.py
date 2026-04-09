"""
Export interactive HTML onset maps: hover shows week index and calendar label.

Uses the same preprocessing as scripts/plot_migration_onset.py (prepare_onset_map_layers).

Run from project root:
    python scripts/plot_onset_interactive.py --species acafly --region north_america
    python scripts/plot_onset_interactive.py --species acafly --backend altair
    python scripts/plot_onset_interactive.py --species acafly --backend plotly

Outputs (per species; spring and fall each get the same pattern):
    Plotly: {species}_spring_onset_interactive.html
    Altair: {species}_spring_onset_interactive_altair.html
  Default --backend both writes Plotly and Altair for each season.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(_scripts_dir))

import plotly.graph_objects as go
import altair as alt

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


def _onset_rect_dataframe(
    onset_ll: np.ndarray,
    date_names: list[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.DataFrame:
    """One row per finite cell with lon/lat band edges for Altair mark_rect."""
    h, w = onset_ll.shape
    dlon = (lon_max - lon_min) / w
    dlat = (lat_max - lat_min) / h
    jj, ii = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
    )
    lon_lo = lon_min + jj * dlon
    lon_hi = lon_lo + dlon
    lat_hi = lat_max - ii * dlat
    lat_lo = lat_hi - dlat
    lon_c = (lon_lo + lon_hi) / 2.0
    lat_c = (lat_lo + lat_hi) / 2.0

    vn = np.isfinite(onset_ll)
    wi = np.round(onset_ll).astype(np.int32)
    n = len(date_names)
    date_lbl = np.array(date_names, dtype=object)[np.clip(wi, 0, n - 1)]
    date_lbl = np.where(vn, date_lbl, "")

    return pd.DataFrame(
        {
            "lon_lo": lon_lo.ravel(),
            "lon_hi": lon_hi.ravel(),
            "lat_lo": lat_lo.ravel(),
            "lat_hi": lat_hi.ravel(),
            "lon": lon_c.ravel(),
            "lat": lat_c.ravel(),
            "week": onset_ll.ravel(),
            "date_label": date_lbl.ravel(),
        }
    )[vn.ravel()].reset_index(drop=True)


def export_onset_plotly_html(
    *,
    layers: dict,
    species: str,
    season: str,
    date_names: list[str],
    output_path: Path,
    title_clean: bool = False,
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
    z_plot = onset_ll
    hover_m = _week_date_hover_strings(onset_ll, date_names)

    if title_clean:
        title = f"{species.upper()} · {season} onset"
    else:
        title = (
            f"{species.upper()} – Migration onset ({season})<br>"
            f"<sup>Blue = earliest, red = latest; {date_start_label}–{date_end_label}</sup>"
        )

    fig = go.Figure(
        data=go.Heatmap(
            x=x_coords,
            y=y_coords,
            z=z_plot,
            zmin=float(week_min_display),
            zmax=float(week_max_display),
            colorscale="Turbo",
            colorbar=dict(title="Onset week index"),
            text=hover_m,
            hovertemplate=(
                "lon=%{x:.4f}<br>lat=%{y:.4f}<br>%{text}<extra></extra>"
            ),
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Longitude", constrain="domain"),
        yaxis=dict(title="Latitude", scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="closest",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  Saved interactive map (Plotly) to {output_path}")


def export_onset_altair_html(
    *,
    layers: dict,
    species: str,
    season: str,
    date_names: list[str],
    output_path: Path,
    title_clean: bool = False,
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

    alt.data_transformers.disable_max_rows()

    df = _onset_rect_dataframe(
        onset_ll, date_names, lon_min, lon_max, lat_min, lat_max
    )
    if df.empty:
        print(f"  Altair: no finite cells, skipping {output_path.name}")
        return

    if title_clean:
        title = alt.TitleParams(text=f"{species.upper()} · {season} onset")
    else:
        title = alt.TitleParams(
            text=f"{species.upper()} – Migration onset ({season})",
            subtitle=f"Turbo: blue = earliest, red = latest · {date_start_label}–{date_end_label}",
        )

    w_px = 800
    h_px = max(200, int(w_px * (lat_max - lat_min) / (lon_max - lon_min)))

    chart = (
        alt.Chart(df)
        .mark_rect(stroke=None)
        .encode(
            x=alt.X("lon_lo:Q", title="Longitude", scale=alt.Scale(domain=[lon_min, lon_max])),
            x2="lon_hi:Q",
            y=alt.Y("lat_lo:Q", title="Latitude", scale=alt.Scale(domain=[lat_min, lat_max], reverse=False)),
            y2="lat_hi:Q",
            color=alt.Color(
                "week:Q",
                title="Onset week",
                scale=alt.Scale(
                    scheme="turbo",
                    domain=[float(week_min_display), float(week_max_display)],
                ),
            ),
            tooltip=[
                alt.Tooltip("lon:Q", format=".4f", title="Lon (center)"),
                alt.Tooltip("lat:Q", format=".4f", title="Lat (center)"),
                alt.Tooltip("week:Q", format=".0f", title="Week index"),
                alt.Tooltip("date_label:N", title="Approx. date"),
            ],
        )
        .properties(title=title, width=w_px, height=h_px)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_path))
    print(f"  Saved interactive map (Altair) to {output_path}")


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
        "--backend",
        choices=["plotly", "altair", "both"],
        default="both",
        help="Interactive library: plotly, altair, or both (default: both)",
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
            if args.backend in ("plotly", "both"):
                export_onset_plotly_html(
                    layers=layers_sp,
                    species=species,
                    season="spring",
                    date_names=date_names,
                    output_path=output_dir / f"{species}_spring_onset_interactive.html",
                    title_clean=args.clean,
                )
            if args.backend in ("altair", "both"):
                export_onset_altair_html(
                    layers=layers_sp,
                    species=species,
                    season="spring",
                    date_names=date_names,
                    output_path=output_dir / f"{species}_spring_onset_interactive_altair.html",
                    title_clean=args.clean,
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
            if args.backend in ("plotly", "both"):
                export_onset_plotly_html(
                    layers=layers_fa,
                    species=species,
                    season="fall",
                    date_names=date_names,
                    output_path=output_dir / f"{species}_fall_onset_interactive.html",
                    title_clean=args.clean,
                )
            if args.backend in ("altair", "both"):
                export_onset_altair_html(
                    layers=layers_fa,
                    species=species,
                    season="fall",
                    date_names=date_names,
                    output_path=output_dir / f"{species}_fall_onset_interactive_altair.html",
                    title_clean=args.clean,
                )

    print(f"\nDone. Interactive HTML saved under {output_dir}")


if __name__ == "__main__":
    main()
