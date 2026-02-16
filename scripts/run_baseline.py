"""
Run baseline Random Forest: load data, extract features, train, evaluate.
Usage: python scripts/run_baseline.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.raster_processing import load_weekly_stack, find_species_data, get_season_dates, load_config
from src.feature_extraction import compute_global_features
from src.labels import build_week_labels
from src.models.random_forest_classifier import build_feature_matrix, train_and_evaluate


def main():
    data_dir = project_root / "data" / "raw"
    species = "yebsap-example"
    resolution = "27km"

    print("Loading weekly abundance...")
    stack, meta = load_weekly_stack(data_dir, species=species, resolution=resolution)
    n_weeks = stack.shape[0]
    print(f"  Shape: {stack.shape}")

    print("Computing movement features...")
    features = compute_global_features(stack)

    print("Building labels from config...")
    species_dir = find_species_data(data_dir, species)
    season_dates = get_season_dates(species_dir)
    config = load_config(species_dir)
    date_names = config.get("DATE_NAMES", [])
    labels, class_names = build_week_labels(date_names, season_dates)
    print(f"  Classes: {class_names}")
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"    {name}: {count} weeks")

    print("Building feature matrix...")
    X = build_feature_matrix(features, n_weeks)
    y = labels
    print(f"  X: {X.shape}, y: {y.shape}")

    print("Training Random Forest (time-series CV)...")
    results = train_and_evaluate(X, y, class_names=class_names, n_splits=5)

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
