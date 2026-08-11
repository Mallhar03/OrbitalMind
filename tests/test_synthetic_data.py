"""
Tests for Iteration 1: Synthetic Data Generator
All tests must pass before moving to Iteration 2.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data

EXPECTED_COLS = ['Timestamp', 'SatelliteID', 'OrbitType', 'ClockError_ns', 'EphemerisError_m']
EXPECTED_ROWS = 6144      # 8 satellites × 768 points
EXPECTED_GEO  = 3
EXPECTED_MEO  = 5


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_gnss_data(seed=42)


def test_row_count(synthetic_df):
    assert len(synthetic_df) == EXPECTED_ROWS, \
        f"Expected {EXPECTED_ROWS} rows, got {len(synthetic_df)}"


def test_columns_exact(synthetic_df):
    assert list(synthetic_df.columns) == EXPECTED_COLS, \
        f"Columns mismatch. Got: {list(synthetic_df.columns)}"


def test_orbit_type_values(synthetic_df):
    unique_types = set(synthetic_df['OrbitType'].unique())
    assert unique_types == {'GEO', 'MEO'}, \
        f"OrbitType must be only GEO or MEO. Got: {unique_types}"


def test_geo_satellite_count(synthetic_df):
    geo_sats = synthetic_df[synthetic_df['OrbitType'] == 'GEO']['SatelliteID'].nunique()
    assert geo_sats == EXPECTED_GEO, \
        f"Expected {EXPECTED_GEO} GEO satellites, got {geo_sats}"


def test_meo_satellite_count(synthetic_df):
    meo_sats = synthetic_df[synthetic_df['OrbitType'] == 'MEO']['SatelliteID'].nunique()
    assert meo_sats == EXPECTED_MEO, \
        f"Expected {EXPECTED_MEO} MEO satellites, got {meo_sats}"


def test_no_nan_values(synthetic_df):
    assert synthetic_df.isnull().sum().sum() == 0, \
        "NaN values found in synthetic data"


def test_clock_error_range(synthetic_df):
    min_val = synthetic_df['ClockError_ns'].min()
    max_val = synthetic_df['ClockError_ns'].max()
    assert min_val >= -20 and max_val <= 20, \
        f"ClockError_ns out of realistic range [-20, 20]. Got [{min_val:.2f}, {max_val:.2f}]"


def test_ephemeris_error_range(synthetic_df):
    min_val = synthetic_df['EphemerisError_m'].min()
    max_val = synthetic_df['EphemerisError_m'].max()
    assert min_val >= -5 and max_val <= 5, \
        f"EphemerisError_m out of realistic range [-5, 5]. Got [{min_val:.2f}, {max_val:.2f}]"


def test_csv_file_exists():
    assert os.path.exists("data/synthetic/gnss_synthetic.csv"), \
        "CSV file not saved to data/synthetic/gnss_synthetic.csv"


def test_timestamp_is_15min_interval(synthetic_df):
    one_sat = synthetic_df[synthetic_df['SatelliteID'] == 'GEO-01'].sort_values('Timestamp')
    diffs = pd.to_datetime(one_sat['Timestamp']).diff().dropna()
    assert all(diffs == pd.Timedelta(minutes=15)), \
        "Timestamps are not 15-minute intervals"


def test_reproducibility():
    df1 = generate_synthetic_gnss_data(seed=42)
    df2 = generate_synthetic_gnss_data(seed=42)
    pd.testing.assert_frame_equal(df1, df2, check_exact=False)
