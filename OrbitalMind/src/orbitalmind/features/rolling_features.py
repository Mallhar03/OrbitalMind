"""Rolling statistical feature creation for GNSS satellite error time series."""
import pandas as pd


def create_rolling_features(
    series: pd.Series,
    windows: list = [4, 8, 16, 96],
) -> pd.DataFrame:
    """
    Create rolling mean, std, min, and max features for a time series.

    Args:
        series: 1-D time series of signal values (trend + periodic)
        windows: list of integer window sizes (in samples)
    Returns:
        DataFrame with 4 × len(windows) columns named 'rolling_{stat}_{W}'.
        First max(windows)-1 rows will contain NaN.
    """
    df = pd.DataFrame(index=series.index)
    for w in windows:
        r = series.rolling(window=w, min_periods=w)
        df[f"rolling_mean_{w}"] = r.mean()
        df[f"rolling_std_{w}"]  = r.std()
        df[f"rolling_min_{w}"]  = r.min()
        df[f"rolling_max_{w}"]  = r.max()
    return df
