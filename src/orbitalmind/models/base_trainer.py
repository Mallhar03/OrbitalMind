"""Shared training utilities and RMSE evaluation for all GNSS models."""
import numpy as np


def compute_rmse_horizons(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE at five standard prediction horizons.

    Horizon-to-step mapping (at 15-minute intervals):
        15min → step 1,  30min → step 2,  1hr → step 4,
        2hr   → step 8,  24hr  → step 96

    When the input arrays are shorter than a given horizon, that horizon uses
    all available elements.

    Args:
        y_true: array of ground-truth values
        y_pred: array of predicted values (same or greater length as y_true)
    Returns:
        Dict with keys '15min', '30min', '1hr', '2hr', '24hr', each a float RMSE.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))

    horizon_steps = {"15min": 1, "30min": 2, "1hr": 4, "2hr": 8, "24hr": 96}
    result = {}
    for key, step in horizon_steps.items():
        k = min(step, n)
        result[key] = float(np.sqrt(np.mean((y_true[:k] - y_pred[:k]) ** 2)))
    return result
