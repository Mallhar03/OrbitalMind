"""
Feature matrix builder for GNSS satellite error prediction.

Combines lag, rolling, FFT, and time-of-day features into a normalised
(X, y) matrix ready for LSTM / TFT / Neural ODE training.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from orbitalmind.features.lag_features import create_lag_features
from orbitalmind.features.rolling_features import create_rolling_features
from orbitalmind.features.fft_features import create_fft_features

_FFT_NAMES  = ["amplitude_24hr", "amplitude_12hr", "spectral_energy", "dominant_freq"]
_SCALER_PATH = "models/saved/scaler.pkl"


def build_feature_matrix(
    preprocessed_data: dict,
    timestamps: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Build a normalised feature matrix from preprocessed satellite data.

    Uses the trend + periodic components (noise excluded) as the signal.
    Drops the first 96 rows where lag/rolling/FFT features are undefined.
    Normalises X to [0, 1] with MinMaxScaler; saves the scaler to disk.

    Args:
        preprocessed_data: output dict from preprocess_satellite()
        timestamps: original 768-length DatetimeIndex (pre-differencing)
    Returns:
        (X, y, feature_names)
        X             : np.ndarray of shape (n_samples, 29)
        y             : np.ndarray of shape (n_samples,)
        feature_names : list of 29 feature name strings
    """
    combined = preprocessed_data["trend"] + preprocessed_data["periodic"]  # (767,)
    series   = pd.Series(combined)

    ts_all  = pd.DatetimeIndex(pd.Series(timestamps).values)
    ts_diff = ts_all[1:]  # 767 timestamps aligned with differenced signal

    lag_df  = create_lag_features(series, lags=[1, 2, 4, 8, 16, 32, 96])
    roll_df = create_rolling_features(series, windows=[4, 8, 16, 96])

    fft_arr = create_fft_features(combined, window_size=96)
    fft_df  = pd.DataFrame(fft_arr, columns=_FFT_NAMES)

    hour_norm = pd.Series(ts_diff).dt.hour / 24.0
    dow_norm  = pd.Series(ts_diff).dt.dayofweek / 6.0
    time_df   = pd.DataFrame({"hour_of_day": hour_norm.values, "day_of_week": dow_norm.values})

    X_df = pd.concat([lag_df, roll_df, fft_df, time_df], axis=1)
    y_s  = series.copy()

    valid_idx = X_df.dropna().index
    X_clean   = X_df.loc[valid_idx].values.astype(float)
    y_clean   = y_s.loc[valid_idx].values.astype(float)

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_clean)

    os.makedirs(os.path.dirname(_SCALER_PATH), exist_ok=True)
    with open(_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled, y_clean, list(X_df.columns)
