"""
Movement metric computation for migration season detection.
Global: centroid displacement, spatial variance.
Local: moving-window metrics (future extension).
"""

import numpy as np
from typing import Optional


def weighted_centroid(arr: np.ndarray, nodata: Optional[float] = np.nan) -> tuple[float, float]:
    """
    Compute weighted centroid (center of mass) of 2D abundance array.
    Uses row/col indices; ignores nodata.
    """
    mask = np.isfinite(arr) & (arr > 0)
    if not np.any(mask):
        return np.nan, np.nan

    rows, cols = np.indices(arr.shape)
    total = np.sum(arr[mask])
    cy = np.sum(rows[mask] * arr[mask]) / total
    cx = np.sum(cols[mask] * arr[mask]) / total
    return float(cy), float(cx)


def spatial_variance(arr: np.ndarray, centroid: Optional[tuple] = None) -> float:
    """
    Weighted spatial variance (spread) around centroid.
    Higher = more dispersed population.
    """
    mask = np.isfinite(arr) & (arr > 0)
    if not np.any(mask):
        return np.nan

    if centroid is None:
        cy, cx = weighted_centroid(arr)
    else:
        cy, cx = centroid

    rows, cols = np.indices(arr.shape)
    weights = arr[mask]
    dy = rows[mask] - cy
    dx = cols[mask] - cx
    var = np.sum(weights * (dy**2 + dx**2)) / np.sum(weights)
    return float(var)


def spatial_entropy(arr: np.ndarray, bins: int = 20) -> float:
    """
    Shannon entropy of abundance distribution (binned).
    Higher = more uniform spread; lower = more concentrated.
    """
    mask = np.isfinite(arr) & (arr > 0)
    if not np.any(mask):
        return np.nan

    vals = arr[mask].ravel()
    probs, _ = np.histogram(vals, bins=bins, density=True)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def centroid_displacement(centroids: list[tuple[float, float]]) -> np.ndarray:
    """
    Displacement between consecutive centroids.
    centroids: list of (row, col) per time step
    Returns: array of length n_timesteps-1
    """
    disp = []
    for i in range(1, len(centroids)):
        cy0, cx0 = centroids[i - 1]
        cy1, cx1 = centroids[i]
        if np.isnan(cy0) or np.isnan(cy1):
            disp.append(np.nan)
        else:
            disp.append(np.sqrt((cy1 - cy0) ** 2 + (cx1 - cx0) ** 2))
    return np.array(disp)


def change_magnitude(stack: np.ndarray) -> np.ndarray:
    """
    Pixel-wise sum of absolute differences between consecutive time steps.
    stack: (n_weeks, height, width)
    Returns: (n_weeks-1,) global change per timestep
    """
    n = stack.shape[0]
    changes = []
    for i in range(1, n):
        diff = np.abs(stack[i].astype(float) - stack[i - 1].astype(float))
        mask = np.isfinite(diff)
        changes.append(np.nansum(diff[mask]))
    return np.array(changes)


def compute_global_features(stack: np.ndarray) -> dict:
    """
    Compute global movement features for each time step.
    stack: (n_weeks, height, width) abundance

    Returns dict with arrays (length n_weeks or n_weeks-1):
        centroid_row, centroid_col
        spatial_variance
        spatial_entropy
        centroid_displacement (n_weeks-1)
        change_magnitude (n_weeks-1)
    """
    n_weeks = stack.shape[0]
    centroids = []
    variances = []
    entropies = []

    for t in range(n_weeks):
        arr = stack[t].astype(float)
        arr[~np.isfinite(arr)] = np.nan
        cy, cx = weighted_centroid(arr)
        centroids.append((cy, cx))
        variances.append(spatial_variance(arr, (cy, cx)))
        entropies.append(spatial_entropy(arr))

    disp = centroid_displacement(centroids)
    chg = change_magnitude(stack)

    return {
        "centroid_row": np.array([c[0] for c in centroids]),
        "centroid_col": np.array([c[1] for c in centroids]),
        "spatial_variance": np.array(variances),
        "spatial_entropy": np.array(entropies),
        "centroid_displacement": disp,
        "change_magnitude": chg,
    }


def compute_local_features(
    stack: np.ndarray,
    cell_size: int = 64,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Compute per-region movement features. Divides raster into grid cells.

    Args:
        stack: (n_weeks, height, width) abundance
        cell_size: pixels per cell (height and width)

    Returns:
        X_local: (n_cells, n_weeks, n_features) - 6 features per cell per week
        grid_shape: (n_rows, n_cols) number of cells
    """
    n_weeks, h, w = stack.shape
    n_rows = max(1, h // cell_size)
    n_cols = max(1, w // cell_size)
    n_cells = n_rows * n_cols

    # Features: centroid_row, centroid_col, variance, entropy, disp, change
    n_feat = 6
    X_local = np.full((n_cells, n_weeks, n_feat), np.nan, dtype=np.float32)

    for ri in range(n_rows):
        for ci in range(n_cols):
            cell_id = ri * n_cols + ci
            r0, r1 = ri * cell_size, min((ri + 1) * cell_size, h)
            c0, c1 = ci * cell_size, min((ci + 1) * cell_size, w)

            centroids = []
            variances = []
            entropies = []

            for t in range(n_weeks):
                arr = stack[t, r0:r1, c0:c1].astype(float)
                arr[~np.isfinite(arr)] = np.nan
                cy, cx = weighted_centroid(arr)
                centroids.append((cy, cx))
                variances.append(spatial_variance(arr, (cy, cx)))
                entropies.append(spatial_entropy(arr))

            disp = centroid_displacement(centroids)
            chg = change_magnitude(stack[:, r0:r1, c0:c1])

            for t in range(n_weeks):
                X_local[cell_id, t, 0] = centroids[t][0] if not np.isnan(centroids[t][0]) else 0
                X_local[cell_id, t, 1] = centroids[t][1] if not np.isnan(centroids[t][1]) else 0
                X_local[cell_id, t, 2] = variances[t] if not np.isnan(variances[t]) else 0
                X_local[cell_id, t, 3] = entropies[t] if not np.isnan(entropies[t]) else 0
                X_local[cell_id, t, 4] = disp[t - 1] if t > 0 and not np.isnan(disp[t - 1]) else 0
                X_local[cell_id, t, 5] = (chg[t - 1] / 1e6) if t > 0 else 0

    return X_local, (n_rows, n_cols)
