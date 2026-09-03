"""
MAD-based outlier removal for GNSS satellite error time series.

Outliers are judged against the LOCAL neighbourhood, not the global level.
A global median/MAD test cannot tell a spike apart from real structure: any
satellite with genuine periodicity has legitimate excursions far from its own
global median, and they score as outliers.

That is not hypothetical. Measured on real 2026-08 data, the previous global
test flagged 82 of J07's 672 ephemeris points (12.2%) and interpolated over
them -- in nine runs, one per calendar day, always in the same ~14:00-17:15
window. It was erasing that satellite's 24-hour periodicity and substituting
fabricated values. J07 is one of only seven GEO satellites in the dataset, and
24-hour periodicity is precisely what the GEO branch exists to model.
"""
import numpy as np
import pandas as pd

# 13 samples at 15-minute cadence is 3.25 hours: far shorter than the 12- and
# 24-hour periodicities that must survive, and far longer than an isolated
# spike. The window tracks real structure so only true outliers stand out.
LOCAL_WINDOW = 13
_MAD_TO_SIGMA = 0.6745   # modified Z-score constant

# On a smooth signal the local MAD collapses towards zero, and dividing by it
# turns negligible residuals into huge scores -- a clean sine had 70 of 672
# points flagged. The scale is therefore floored at a fraction of the series'
# own variability: a deviation smaller than a tenth of the typical variation is
# not an outlier under any useful definition.
_SCALE_FLOOR_FRAC = 0.1


def remove_outliers_mad(
    series: pd.Series,
    threshold: float = 3.5,
    window: int = LOCAL_WINDOW,
) -> pd.Series:
    """
    Replace local outliers with linearly interpolated values.

    Uses a Hampel-style test: each point is compared against the median of its
    own neighbourhood, scaled by the local median absolute deviation, so smooth
    periodic structure is preserved and only isolated spikes are removed.

    Args:
        series:    time series of satellite error values
        threshold: modified Z-score cutoff (default 3.5)
        window:    number of samples in the centred local window
    Returns:
        Series with outliers replaced by linear interpolation; same index as input.
    """
    s = series.copy().astype(float)
    if len(s) < 2:
        return s

    # Too short for a meaningful local window: fall back to the global test.
    if len(s) < window * 2:
        centre = np.median(s)
        spread = np.median(np.abs(s - centre))
        modified_z = _MAD_TO_SIGMA * (s - centre) / (spread + 1e-8)
    else:
        min_periods = max(3, window // 3)
        local_median = s.rolling(window, center=True, min_periods=min_periods).median()
        residual = s - local_median
        local_mad = (residual.abs()
                       .rolling(window, center=True, min_periods=min_periods)
                       .median())

        global_mad = float(np.median(np.abs(s - np.median(s))))
        scale = local_mad.clip(lower=_SCALE_FLOOR_FRAC * global_mad)
        modified_z = _MAD_TO_SIGMA * residual / (scale + 1e-8)

    outliers = modified_z.abs() > threshold
    s[outliers] = np.nan
    return s.interpolate(method="linear", limit_direction="both")
