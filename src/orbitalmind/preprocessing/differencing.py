"""Single-difference transformation for making GNSS error series stationary."""
import numpy as np
import pandas as pd


def single_difference(series: pd.Series) -> tuple[pd.Series, float]:
    """
    Apply first-order differencing to produce a stationary series.

    Args:
        series: cleaned satellite error time series (n points)
    Returns:
        (differenced, first_value) where differenced has n-1 points and
        first_value is the original series[0] needed for reconstruction.
    """
    first_value = float(series.iloc[0])
    differenced = series.diff().dropna()
    return differenced, first_value


def reverse_single_difference(differenced: np.ndarray, first_value: float) -> np.ndarray:
    """
    Reconstruct original-scale series from a differenced array.

    Args:
        differenced: first-difference array (n-1 values)
        first_value: the original first observation before differencing
    Returns:
        Reconstructed array of length n (n-1 differences + 1 initial value).
    """
    return np.concatenate([[first_value], first_value + np.cumsum(differenced)])
