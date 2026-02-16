"""
Run baseline: load data, extract features, train, evaluate.

Phases:
  Phase 1 (default): Binary movement vs no-movement (global)
  Phase 1 (--4class): 4-class season (global)
  Phase 2 (--regional): Per-region prediction with local features

Usage:
  python scripts/run_baseline.py           # Phase 1 binary
  python scripts/run_baseline.py --4class   # Phase 1 four-class
  python scripts/run_baseline.py --regional # Phase 2 per-region
  python scripts/run_baseline.py --regional --binary  # Phase 2 regional + binary
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.raster_processing import load_weekly_stack, find_species_data, get_season_dates, load_config
from src.feature_extraction import compute_global_features, compute_local_features
from src.labels import build_week_labels, build_binary_labels
from src.models.random_forest_classifier import (
    build_feature_matrix,
    build_regional_feature_matrix,
    train_and_evaluate,
)


def run_phase1(X: np.ndarray, y: np.ndarray, class_names: list[str], mode: str) -> dict:
    """Phase 1: global prediction."""
    print(f"Training Random Forest ({mode} labels, time-series CV)...")
    return train_and_evaluate(X, y, class_names=class_names, n_splits=5)


def run_phase2(
    X_local: np.ndarray,
    labels: np.ndarray,
    n_weeks: int,
    class_names: list[str],
    grid_shape: tuple[int, int],
) -> dict:
    """Phase 2: per-region prediction with local features."""
    X_flat, y_flat = build_regional_feature_matrix(X_local, labels, n_weeks)
    n_cells = X_local.shape[0]
    print(f"  Regional samples: {X_flat.shape[0]} ({n_cells} cells x {n_weeks} weeks)")
    print(f"  Grid: {grid_shape[0]} x {grid_shape[1]} cells")
    return train_and_evaluate(X_flat, y_flat, class_names=class_names, n_splits=5)


def main():
    parser = argparse.ArgumentParser(description="Run migration season baseline")
    parser.add_argument("--4class", action="store_true", help="Use 4-class labels (default: binary)")
    parser.add_argument("--regional", action="store_true", help="Phase 2: per-region with local features")
    parser.add_argument("--binary", action="store_true", help="Use binary labels (with --regional)")
    parser.add_argument("--cell-size", type=int, default=64, help="Cell size for regional (default: 64)")
    args = parser.parse_args()

    use_binary = (not args.__dict__["4class"]) or args.binary
    use_regional = args.regional

    data_dir = project_root / "data" / "raw"
    species = "yebsap-example"
    resolution = "27km"

    print("Loading weekly abundance...")
    stack, meta = load_weekly_stack(data_dir, species=species, resolution=resolution)
    n_weeks = stack.shape[0]
    print(f"  Shape: {stack.shape}")

    species_dir = find_species_data(data_dir, species)
    season_dates = get_season_dates(species_dir)
    config = load_config(species_dir)
    date_names = config.get("DATE_NAMES", [])

    if use_binary:
        labels, class_names = build_binary_labels(date_names, season_dates)
    else:
        labels, class_names = build_week_labels(date_names, season_dates)

    print(f"  Labels: {class_names}")
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"    {name}: {count} weeks")

    if use_regional:
        print("Computing local features (per-region)...")
        X_local, grid_shape = compute_local_features(stack, cell_size=args.cell_size)
        print("Building regional feature matrix...")
        X = None  # not used
        results = run_phase2(X_local, labels, n_weeks, class_names, grid_shape)
    else:
        print("Computing global movement features...")
        features = compute_global_features(stack)
        print("Building feature matrix...")
        X = build_feature_matrix(features, n_weeks)
        y = labels
        print(f"  X: {X.shape}, y: {y.shape}")
        mode = "binary" if use_binary else "4-class"
        results = run_phase1(X, y, class_names, mode)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} (+/- {results['cv_accuracy_std']:.3f})")
    print(f"Train Accuracy: {results['train_accuracy']:.3f}")
    print("\nClassification Report:")
    print(results["classification_report"])
    print("Confusion Matrix (rows=actual, cols=pred):")
    print("  ", class_names)
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()
