"""IOD (Issue of Data) jump correction for GNSS satellite error time series."""
import numpy as np
import pandas as pd


def correct_iod_jumps(series: pd.Series, threshold_ns: float = 2.0) -> pd.Series:
    """
    Detect and remove IOD upload discontinuities from a satellite error series.

    At each timestep where |diff| > threshold, the jump offset is subtracted
    from all subsequent values, making the series continuous.

    Args:
        series: satellite error time series (clock in ns or ephemeris in m)
        threshold_ns: jump magnitude threshold (use 2.0 for ns, 1.0 for metres)
    Returns:
        Corrected series with the same index as input.
    """
    s = series.copy().astype(float).reset_index(drop=True)
    diffs = s.diff().abs()
    jump_indices = diffs[diffs > threshold_ns].index.tolist()
    for idx in jump_indices:
        offset = s.iloc[idx] - s.iloc[idx - 1]
        s.iloc[idx:] -= offset
    s.index = series.index
    return s
