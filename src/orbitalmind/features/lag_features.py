"""Lag feature creation for GNSS satellite error time series."""
import pandas as pd


def create_lag_features(
    series: pd.Series,
    lags: list = [1, 2, 4, 8, 16, 32, 96],
) -> pd.DataFrame:
    """
    Create lag (auto-regressive) features for a time series.

    Args:
        series: 1-D time series of signal values (trend + periodic)
        lags: list of integer lag values (steps back in time)
    Returns:
        DataFrame with one column per lag, named 'lag_N'.
        First max(lags) rows will contain NaN.
    """
    df = pd.DataFrame(index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df
