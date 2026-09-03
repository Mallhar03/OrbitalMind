"""Preprocessing pipeline orchestrating all 4 steps for a single satellite/error pair."""
import numpy as np
import pandas as pd

from orbitalmind.preprocessing.outlier_removal import remove_outliers_mad
from orbitalmind.preprocessing.iod_correction import correct_iod_jumps
from orbitalmind.preprocessing.detrending import detrend_signal
from orbitalmind.preprocessing.decomposition import decompose_signal

def preprocess_satellite(
    df: pd.DataFrame,
    sat_id: str,
    error_col: str,
) -> dict:
    """
    Run the full 4-step preprocessing pipeline for one satellite and error column.

    Steps: MAD outlier removal → IOD jump correction → linear detrending → EMD decomposition.

    Args:
        df: raw DataFrame with columns [Timestamp, SatelliteID, OrbitType,
            ClockError_ns, EphemerisError_m]
        sat_id: satellite identifier (e.g. 'GEO-01', 'MEO-03')
        error_col: column to process ('ClockError_ns' or 'EphemerisError_m')
    Two original-scale series are returned and they are not interchangeable:

        observed          outlier-cleaned, still in the MEASUREMENT frame.
                          Anchor forecast reconstruction on this.
        original_cleaned  additionally IOD-corrected, so it is jump-free and
                          suitable for modelling, but it sits in a shifted
                          frame whenever any jump was removed. On real IGS
                          data that shift reached 3339 ns (G02) and 18216 ns
                          (G03); anchoring on it puts the submission that far
                          out.

    Returns:
        Dict with keys:
            sat_id, error_col, trend, periodic, noise,
            slope, intercept, observed, original_cleaned, timestamps
    """
    sat_df = df[df["SatelliteID"] == sat_id].sort_values("Timestamp").copy()
    sat_df["Timestamp"] = pd.to_datetime(sat_df["Timestamp"], format="mixed")
    sat_df = sat_df.drop_duplicates(subset=["Timestamp"]).set_index("Timestamp")
    
    # Interpolate to a uniform 15-minute grid
    sat_df = sat_df.resample("15min").interpolate(method="time")
    
    raw_series = sat_df[error_col].reset_index(drop=True)
    timestamps  = pd.to_datetime(sat_df.index.values)

    observed = remove_outliers_mad(raw_series)
    # Threshold is derived per satellite; a fixed one flagged every step of the
    # fast-drifting clocks as a jump and removed their entire trend.
    cleaned = correct_iod_jumps(observed)

    detrended, slope, intercept = detrend_signal(cleaned)
    detrended_values = detrended.values.astype(float)

    trend, periodic, noise = decompose_signal(detrended_values)

    return {
        "sat_id":           sat_id,
        "error_col":        error_col,
        "trend":            trend,
        "periodic":         periodic,
        "noise":            noise,
        "slope":            slope,
        "intercept":        intercept,
        "observed":         observed.values.astype(float),
        "original_cleaned": cleaned.values.astype(float),
        "timestamps":       timestamps,  # full length preserved
    }
