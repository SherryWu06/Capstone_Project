"""
Random Forest baseline for migration season classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def build_feature_matrix(features: dict, n_weeks: int) -> np.ndarray:
    """
    Build X matrix (n_weeks, n_features). Aligns displacement/change to n_weeks.
    """
    # Per-week features
    centroid_row = features["centroid_row"]
    centroid_col = features["centroid_col"]
    spatial_variance = features["spatial_variance"]
    spatial_entropy = features["spatial_entropy"]

    # Lagged (n_weeks-1) - pad first week with 0
    disp = np.concatenate([[0], features["centroid_displacement"]])
    chg = np.concatenate([[0], features["change_magnitude"]])

    X = np.column_stack([
        centroid_row,
        centroid_col,
        spatial_variance,
        spatial_entropy,
        disp,
        chg / 1e6,  # scale
    ])
    return X


def build_regional_feature_matrix(
    X_local: np.ndarray,
    labels: np.ndarray,
    n_weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten per-region features for classification.
    X_local: (n_cells, n_weeks, n_features)
    labels: (n_weeks,) global labels
    Returns X, y ordered by week (for time-series CV).
    """
    n_cells, _, n_feat = X_local.shape
    # Order by week: week 0 (all cells), week 1 (all cells), ...
    X_flat = X_local.transpose(1, 0, 2).reshape(n_weeks * n_cells, n_feat)
    y_flat = np.repeat(labels, n_cells)
    return X_flat, y_flat


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train RF with time-series CV, return metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

    accs = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)
        accs.append(accuracy_score(y_test, pred))

    # Final fit on all data for reporting
    X_s = scaler.fit_transform(X)
    clf.fit(X_s, y)
    y_pred = clf.predict(X_s)

    target_names = class_names or [str(i) for i in range(len(np.unique(y)))]

    return {
        "cv_accuracy_mean": np.mean(accs),
        "cv_accuracy_std": np.std(accs),
        "train_accuracy": accuracy_score(y, y_pred),
        "classification_report": classification_report(y, y_pred, target_names=target_names),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "model": clf,
        "scaler": scaler,
    }


def train_and_evaluate_species_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
    random_state: int = 42,
) -> dict:
    """
    Train on train species, evaluate on held-out test species.
    Tests generalization to unseen species.
    """
    scaler = StandardScaler()
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf.fit(X_train_s, y_train)
    y_train_pred = clf.predict(X_train_s)
    y_test_pred = clf.predict(X_test_s)

    target_names = class_names or [str(i) for i in range(len(np.unique(np.concatenate([y_train, y_test]))))]

    return {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "classification_report_train": classification_report(y_train, y_train_pred, target_names=target_names),
        "classification_report_test": classification_report(y_test, y_test_pred, target_names=target_names),
        "confusion_matrix_test": confusion_matrix(y_test, y_test_pred),
        "model": clf,
        "scaler": scaler,
    }
