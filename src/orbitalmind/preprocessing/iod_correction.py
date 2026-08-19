"""
IOD (Issue of Data) jump correction for GNSS satellite error time series.

An IOD upload shows up as a step discontinuity in the broadcast-minus-precise
error. Detecting it means asking whether a step is anomalous *for this
satellite*, which a fixed threshold cannot do: GPS clock drift rates vary by
orders of magnitude across the constellation.

The original fixed 2.0 ns threshold classified every single step of the fast
satellites as a jump -- 671 of 671 for both G02 and G03 in the real IGS data --
and subtracting all of those offsets removed the entire trend, leaving the
models nothing to predict. The threshold is now derived from the robust spread
of the satellite's own first differences.
"""
import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826   # scales MAD to a standard-deviation equivalent
JUMP_SIGMAS  = 8.0      # a jump must exceed this many robust sigmas
MIN_SPREAD   = 1e-9


def _jump_threshold(diffs: pd.Series, threshold_ns: float | None) -> float:
    """
    Choose the discontinuity threshold for one satellite.

    Args:
        diffs:        first differences of the series
        threshold_ns: explicit override, or None to derive one
    Returns:
        Absolute step size above which a difference counts as an IOD jump.
    """
    if threshold_ns is not None:
        return float(threshold_ns)

    finite = diffs.dropna().astype(float)
    if len(finite) == 0:
        return float("inf")

    centre = float(np.median(finite))
    mad    = float(np.median(np.abs(finite - centre)))
    spread = mad * MAD_TO_SIGMA

    if spread < MIN_SPREAD:
        # A perfectly regular series: only a genuine break stands out at all.
        span = float(np.max(np.abs(finite - centre)))
        return float("inf") if span < MIN_SPREAD else max(span * 0.5, MIN_SPREAD)

    return abs(centre) + JUMP_SIGMAS * spread


def count_jumps(series: pd.Series, threshold_ns: float | None = None) -> int:
    """
    Count IOD discontinuities without modifying the series.

    Args:
        series:       satellite error time series
        threshold_ns: explicit threshold, or None to derive one
    Returns:
        Number of steps classified as IOD jumps.
    """
    s = pd.Series(series).astype(float).reset_index(drop=True)
    diffs = s.diff()
    threshold = _jump_threshold(diffs, threshold_ns)
    return int((diffs.abs() > threshold).sum())


def correct_iod_jumps(
    series: pd.Series,
    threshold_ns: float | None = None,
) -> pd.Series:
    """
    Detect and remove IOD upload discontinuities from a satellite error series.

    At each step whose magnitude is anomalous for this satellite, the offset is
    subtracted from all subsequent values, making the series continuous. Normal
    drift -- however fast -- is left untouched.

    Note that the result lives in a shifted coordinate frame whenever any jump
    is corrected. Callers reconstructing forecasts to original units must
    anchor on the observed series, not on this one; see
    preprocess_satellite()'s 'observed' output.

    Args:
        series:       satellite error time series (clock in ns, ephemeris in m)
        threshold_ns: explicit jump threshold. None (the default) derives one
                      from the robust spread of this satellite's own steps.
    Returns:
        Corrected series with the same index as the input.
    """
    s = pd.Series(series).copy().astype(float).reset_index(drop=True)
    diffs = s.diff()
    threshold = _jump_threshold(diffs, threshold_ns)

    jump_indices = diffs[diffs.abs() > threshold].index.tolist()
    for idx in jump_indices:
        offset = s.iloc[idx] - s.iloc[idx - 1]
        s.iloc[idx:] -= offset

    s.index = pd.Series(series).index
    return s
