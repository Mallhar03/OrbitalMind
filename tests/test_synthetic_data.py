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
    """
    Ephemeris error must be physically plausible and, crucially, unsaturated.

    This test previously asserted the range [-5, 5] m. That bound held only
    because the generator clipped to it, so the test was encoding the clipping
    rather than the physics -- and the clipping was a real defect: it pinned 801
    GEO rows, 34.8% of all GEO data, at exactly +5.000 m. Persistence then scored
    0.000 against them and the model scored up to 65 m, so those rows measured a
    clipping artifact instead of forecasting skill.

    The bound is now set from physics rather than from the old clip: broadcast
    orbit errors run to a few metres for MEO and larger for geosynchronous
    satellites, while anything beyond tens of metres would indicate a generator
    fault. The saturation check below is the stronger assertion, and is what
    would actually have caught the original defect.
    """
    values = synthetic_df['EphemerisError_m']
    min_val, max_val = values.min(), values.max()

    assert -50 < min_val and max_val < 50, \
        f"EphemerisError_m implausible for any GNSS orbit: [{min_val:.2f}, {max_val:.2f}]"

    # MEO orbits are better determined than geosynchronous ones.
    meo = synthetic_df[synthetic_df['OrbitType'] == 'MEO']['EphemerisError_m']
    assert meo.abs().max() < 15, \
        f"MEO ephemeris error {meo.abs().max():.2f} m is too large to be realistic"

    # The real guard: no value may pile up on a boundary. Saturated values carry
    # no information, and a model trained on them learns a ceiling that does not
    # exist in the real signal.
    for bound in (min_val, max_val):
        pinned = int((values == bound).sum())
        assert pinned < 10, (
            f"{pinned} rows sit at exactly {bound:.3f} m — the distribution is "
            f"saturating, which is the clipping defect returning"
        )


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
