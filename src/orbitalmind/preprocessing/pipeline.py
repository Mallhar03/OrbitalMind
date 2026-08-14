"""Preprocessing pipeline orchestrating all 4 steps for a single satellite/error pair."""
import numpy as np
import pandas as pd

from orbitalmind.preprocessing.outlier_removal import remove_outliers_mad
from orbitalmind.preprocessing.iod_correction import correct_iod_jumps
from orbitalmind.preprocessing.differencing import single_difference
from orbitalmind.preprocessing.decomposition import decompose_signal

# IOD jump thresholds per error type
_IOD_THRESHOLDS = {
    "ClockError_ns":    2.0,
    "EphemerisError_m": 1.0,
}


def preprocess_satellite(
    df: pd.DataFrame,
    sat_id: str,
    error_col: str,
) -> dict:
    """
    Run the full 4-step preprocessing pipeline for one satellite and error column.

    Steps: MAD outlier removal → IOD jump correction → single differencing → EMD decomposition.

    Args:
        df: raw DataFrame with columns [Timestamp, SatelliteID, OrbitType,
            ClockError_ns, EphemerisError_m]
        sat_id: satellite identifier (e.g. 'GEO-01', 'MEO-03')
        error_col: column to process ('ClockError_ns' or 'EphemerisError_m')
    Returns:
        Dict with keys:
            sat_id, error_col, trend, periodic, noise,
            first_value, original_cleaned, timestamps
    """
    sat_df = df[df["SatelliteID"] == sat_id].sort_values("Timestamp").copy()
    raw_series = sat_df[error_col].reset_index(drop=True)
    timestamps  = pd.to_datetime(sat_df["Timestamp"].values)

    iod_threshold = _IOD_THRESHOLDS.get(error_col, 2.0)

    cleaned = remove_outliers_mad(raw_series)
    cleaned = correct_iod_jumps(cleaned, threshold_ns=iod_threshold)

    differenced, first_value = single_difference(cleaned)
    diff_values = differenced.values.astype(float)

    trend, periodic, noise = decompose_signal(diff_values)

    return {
        "sat_id":           sat_id,
        "error_col":        error_col,
        "trend":            trend,
        "periodic":         periodic,
        "noise":            noise,
        "first_value":      first_value,
        "original_cleaned": cleaned.values.astype(float),
        "timestamps":       timestamps[1:],  # aligned with differenced length
    }
