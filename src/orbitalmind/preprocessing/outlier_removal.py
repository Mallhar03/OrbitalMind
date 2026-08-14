"""MAD-based outlier removal for GNSS satellite error time series."""
import numpy as np
import pandas as pd


def remove_outliers_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Replace outliers detected via Modified Z-Score (MAD) with linearly interpolated values.

    Args:
        series: time series of satellite error values
        threshold: modified Z-score cutoff (default 3.5)
    Returns:
        Series with outliers replaced by linear interpolation; same index as input.
    """
    s = series.copy().astype(float)
    median = np.median(s)
    mad = np.median(np.abs(s - median))
    modified_z = 0.6745 * (s - median) / (mad + 1e-8)
    outliers = np.abs(modified_z) > threshold
    s[outliers] = np.nan
    s = s.interpolate(method="linear", limit_direction="both")
    return s
