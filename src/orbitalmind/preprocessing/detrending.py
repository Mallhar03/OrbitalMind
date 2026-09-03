"""Linear detrending transformation for making GNSS error series stationary."""

import numpy as np
import pandas as pd

def detrend_signal(series: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Apply global linear detrending to produce a stationary series.

    Unlike differencing which consumes one sample and accumulates drift when
    reconstructed, detrending centers the signal and allows absolute
    reconstruction without compounding variance.

    Args:
        series: 1-D pandas Series (original scale errors, length n)
    Returns:
        (detrended, slope, intercept) where detrended has n points.
    """
    t = np.arange(len(series), dtype=np.float64)
    y = series.values.astype(np.float64)
    
    # Fit linear polynomial: y = m*t + c
    slope, intercept = np.polyfit(t, y, 1)
    
    # Evaluate line and subtract
    line = slope * t + intercept
    detrended_values = y - line
    
    detrended_series = pd.Series(detrended_values, index=series.index)
    return detrended_series, float(slope), float(intercept)
