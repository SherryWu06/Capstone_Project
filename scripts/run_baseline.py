"""
Run baseline: load data, extract features, train, evaluate.

Phases:
  Phase 1 (default): Binary movement vs no-movement (global)
  Phase 1 (--4class): 4-class season (global)
  Phase 2 (--regional): Per-region prediction with local features

Data sources:
  ebirdst (default): data/raw/2023/yebsap-example/
  matt: data/raw/Matt/ (requires data/labels/matt_species_seasons.json)
  all: combine ebirdst + all Matt species with data

Usage:
  python scripts/run_baseline.py              # Phase 1 binary, ebirdst
  python scripts/run_baseline.py --matt       # Matt data (acafly or --species)
  python scripts/run_baseline.py --matt --species acafly
  python scripts/run_baseline.py --source all # Multi-species
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.raster_processing import (
    load_weekly_stack,
    load_matt_stack,
    list_matt_species,
    find_species_data,
    get_season_dates,
    load_config,
)
from src.feature_extraction import compute_global_features, compute_local_features
from src.labels import build_week_labels, build_binary_labels, get_season_dates_from_json
from src.models.random_forest_classifier import (
    build_feature_matrix,
    build_regional_feature_matrix,
    train_and_evaluate,
    train_and_evaluate_species_split,
)
from src.models.mil_classifier import (
    bags_from_regional,
    train_and_evaluate_mil,
    train_and_evaluate_mil_cv,
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


def load_species_data(data_dir: Path, source: str, species: str, resolution: str, year: int):
    """Load stack and labels for one species. Returns (stack, meta, labels, class_names) or None."""
    if source == "ebirdst":
        stack, meta = load_weekly_stack(data_dir, species=species, resolution=resolution, year=year)
        species_dir = find_species_data(data_dir, species, year)
        season_dates = get_season_dates(species_dir)
        config = load_config(species_dir)
        date_names = config.get("DATE_NAMES", [])
    else:  # matt
        stack, meta = load_matt_stack(data_dir, species=species, resolution=resolution, year=year)
        labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Run 'Rscript scripts/export_season_dates.R' to create {labels_path}"
            )
        season_dates, date_names = get_season_dates_from_json(labels_path, species, year=year)
    return stack, meta, season_dates, date_names


def main():
    parser = argparse.ArgumentParser(description="Run migration season baseline")
    parser.add_argument("--4class", action="store_true", help="Use 4-class labels (default: binary)")
    parser.add_argument("--regional", action="store_true", help="Phase 2: per-region with local features")
    parser.add_argument("--binary", action="store_true", help="Use binary labels (with --regional)")
    parser.add_argument("--cell-size", type=int, default=64, help="Cell size for regional (default: 64)")
    parser.add_argument(
        "--source",
        choices=["ebirdst", "matt", "all"],
        default="ebirdst",
        help="Data source (default: ebirdst)",
    )
    parser.add_argument("--matt", action="store_true", help="Shortcut for --source matt")
    parser.add_argument("--species", type=str, help="Species code (e.g. acafly for Matt)")
    parser.add_argument("--resolution", type=str, default="27km", help="Resolution (default: 27km)")
    parser.add_argument(
        "--species-split",
        action="store_true",
        help="Train/test split by species (for Matt/all). Tests generalization to unseen species.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of species for test set when using --species-split (default: 0.2)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for species split (default: 42)")
    args = parser.parse_args()

    use_binary = (not args.__dict__["4class"]) or args.binary
    use_regional = args.regional
    source = "matt" if args.matt else args.source
    data_dir = project_root / "data" / "raw"

    # Determine species to load
    use_species_split = args.species_split
    labels_path = project_root / "data" / "labels" / "matt_species_seasons.json"
    json_species = set()
    if labels_path.exists():
        with open(labels_path) as f:
            json_species = set(json.load(f).keys())

    if source == "all":
        all_species = []
        for src, sp, yr in [("ebirdst", "yebsap-example", 2023)]:
            all_species.append((src, sp, yr))
        matt_species = list_matt_species(data_dir, year=2024)
        for sp in matt_species:
            if sp in json_species:
                all_species.append(("matt", sp, 2024))
    elif source == "matt":
        if args.species:
            all_species = [("matt", args.species, 2024)]
        elif use_species_split:
            # Use all Matt species for species-level split
            matt_species = list_matt_species(data_dir, year=2024)
            all_species = [
                ("matt", sp, 2024) for sp in matt_species if sp in json_species
            ]
        else:
            matt_species = list_matt_species(data_dir, year=2024)
            sp = matt_species[0] if matt_species else "acafly"
            all_species = [("matt", sp, 2024)]
    else:
        all_species = [("ebirdst", "yebsap-example", 2023)]

    # Species-level train/test split (requires 2+ species)
    if use_species_split and len(all_species) >= 2:
        train_species, test_species = train_test_split(
            all_species,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"  Species split: {len(train_species)} train, {len(test_species)} test")
        print(f"  Train species: {[s[1] for s in train_species]}")
        print(f"  Test species: {[s[1] for s in test_species]}")
    else:
        train_species = all_species
        test_species = []
        if use_species_split and len(all_species) < 2:
            print("  Warning: --species-split requires 2+ species. Using standard CV.")

    # Load train species
    stacks_train, all_labels_train, bags_train, class_names = [], [], [], None
    for src, species, year in train_species:
        try:
            out = load_species_data(data_dir, src, species, args.resolution, year)
            stack, meta, season_dates, date_names = out
            stack = stack.astype(np.float32)
            if use_binary:
                labels, class_names = build_binary_labels(date_names, season_dates, year)
            else:
                labels, class_names = build_week_labels(date_names, season_dates, year)

            if use_regional:
                features = compute_local_features(stack, cell_size=args.cell_size)[0]
                n_weeks = stack.shape[0]
                bags, bag_labels = bags_from_regional(features, labels)
                bags_train.extend(bags)
                all_labels_train.extend(bag_labels)
            else:
                features = compute_global_features(stack)
                X = build_feature_matrix(features, stack.shape[0])
                stacks_train.append(X)
                all_labels_train.append(labels)
            print(f"  Loaded {species} ({src}): {stack.shape[0]} weeks [train]")
        except FileNotFoundError as e:
            print(f"  Skip {species}: {e}")
            continue

    # Load test species (if species split)
    stacks_test, all_labels_test, bags_test = [], [], []
    for src, species, year in test_species:
        try:
            out = load_species_data(data_dir, src, species, args.resolution, year)
            stack, meta, season_dates, date_names = out
            stack = stack.astype(np.float32)
            if use_binary:
                labels, _ = build_binary_labels(date_names, season_dates, year)
            else:
                labels, _ = build_week_labels(date_names, season_dates, year)

            if use_regional:
                features = compute_local_features(stack, cell_size=args.cell_size)[0]
                bags, bag_labels = bags_from_regional(features, labels)
                bags_test.extend(bags)
                all_labels_test.extend(bag_labels)
            else:
                features = compute_global_features(stack)
                X = build_feature_matrix(features, stack.shape[0])
                stacks_test.append(X)
                all_labels_test.append(labels)
            print(f"  Loaded {species} ({src}): {stack.shape[0]} weeks [test]")
        except FileNotFoundError as e:
            print(f"  Skip {species}: {e}")
            continue

    if not stacks_train and not bags_train:
        print("No data loaded. Check paths and run export_season_dates.R for Matt.")
        return

    if use_regional:
        y_train = np.array(all_labels_train)
        use_species_split_eval = bool(test_species and bags_test)
        print(f"  Train: {len(bags_train)} bags (weeks), y {y_train.shape}")
        if use_species_split_eval:
            y_test = np.array(all_labels_test)
            print(f"  Test:  {len(bags_test)} bags (weeks), y {y_test.shape}")
    else:
        X_train = np.vstack(stacks_train)
        y_train = np.concatenate(all_labels_train)
        print(f"  Train: X {X_train.shape}, y {y_train.shape}")

        if test_species and stacks_test:
            X_test = np.vstack(stacks_test)
            y_test = np.concatenate(all_labels_test)
            print(f"  Test:  X {X_test.shape}, y {y_test.shape}")
            use_species_split_eval = True
        else:
            X_train, y_train, use_species_split_eval = X_train, y_train, False

    print(f"  Labels: {class_names}")
    for i, name in enumerate(class_names):
        count = np.sum(y_train == i)
        print(f"    {name}: {count} samples (train)")

    if use_regional:
        print("Training Attention MIL (semi-supervised: bag=week, instances=cells)...")
        if use_species_split_eval:
            results = train_and_evaluate_mil(
                bags_train, y_train,
                bags_test, y_test,
                class_names=class_names,
                n_epochs=80,
                random_state=args.random_state,
            )
            print("\n" + "=" * 50)
            print("RESULTS (MIL, species-level holdout)")
            print("=" * 50)
            print(f"Train Accuracy: {results['train_accuracy']:.3f}")
            print(f"Test Accuracy (unseen species): {results['test_accuracy']:.3f}")
            print("\nClassification Report (test set):")
            print(results["classification_report_test"])
            print("Confusion Matrix (test, rows=actual, cols=pred):")
            print("  ", class_names)
            print(results["confusion_matrix_test"])
        else:
            results = train_and_evaluate_mil_cv(
                bags_train, y_train,
                class_names=class_names,
                n_splits=5,
                n_epochs=80,
                random_state=args.random_state,
            )
            print("\n" + "=" * 50)
            print("RESULTS (MIL, time-series CV)")
            print("=" * 50)
            print(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} (+/- {results['cv_accuracy_std']:.3f})")
            print(f"Train Accuracy: {results['train_accuracy']:.3f}")
            print("\nClassification Report:")
            print(results["classification_report"])
    elif use_species_split_eval:
        print("Training Random Forest (species split: train on train species, eval on test species)...")
        results = train_and_evaluate_species_split(
            X_train, y_train, X_test, y_test,
            class_names=class_names,
            random_state=args.random_state,
        )
        print("\n" + "=" * 50)
        print("RESULTS (species-level holdout)")
        print("=" * 50)
        print(f"Train Accuracy: {results['train_accuracy']:.3f}")
        print(f"Test Accuracy (unseen species): {results['test_accuracy']:.3f}")
        print("\nClassification Report (test set):")
        print(results["classification_report_test"])
        print("Confusion Matrix (test, rows=actual, cols=pred):")
        print("  ", class_names)
        print(results["confusion_matrix_test"])
    else:
        X, y = X_train, y_train
        mode = "binary" if use_binary else "4-class"
        print(f"Training Random Forest ({mode}, time-series CV)...")
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
