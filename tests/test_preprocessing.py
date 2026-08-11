"""
Tests for Iteration 2: Preprocessing Pipeline
All tests must pass before moving to Iteration 3.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite

REQUIRED_KEYS = ['sat_id', 'error_col', 'trend', 'periodic', 'noise',
                 'first_value', 'original_cleaned', 'timestamps']


@pytest.fixture(scope="module")
def sample_preprocessed():
    df = generate_synthetic_gnss_data(seed=42)
    return preprocess_satellite(df, 'GEO-01', 'ClockError_ns')


@pytest.fixture(scope="module")
def sample_preprocessed_meo():
    df = generate_synthetic_gnss_data(seed=42)
    return preprocess_satellite(df, 'MEO-01', 'ClockError_ns')


def test_output_keys(sample_preprocessed):
    for key in REQUIRED_KEYS:
        assert key in sample_preprocessed, f"Missing key: {key}"


def test_no_nan_in_components(sample_preprocessed):
    for key in ['trend', 'periodic', 'noise']:
        arr = sample_preprocessed[key]
        assert not np.any(np.isnan(arr)), f"NaN found in {key}"


def test_signal_reconstruction(sample_preprocessed):
    trend    = sample_preprocessed['trend']
    periodic = sample_preprocessed['periodic']
    noise    = sample_preprocessed['noise']
    reconstructed = trend + periodic + noise
    original_diff = np.diff(sample_preprocessed['original_cleaned'])
    min_len = min(len(reconstructed), len(original_diff))
    error = np.max(np.abs(reconstructed[:min_len] - original_diff[:min_len]))
    assert error < 1e-4, f"Reconstruction error too large: {error}"


def test_length_after_differencing(sample_preprocessed):
    original_len = 768  # 8 days × 96
    trend_len = len(sample_preprocessed['trend'])
    assert trend_len == original_len - 1, \
        f"After differencing, length should be {original_len-1}, got {trend_len}"


def test_iod_correction_reduces_jumps(sample_preprocessed):
    df = generate_synthetic_gnss_data(seed=42)
    sat_df = df[df['SatelliteID'] == 'GEO-01']['ClockError_ns']
    original_max_jump = sat_df.diff().abs().max()
    cleaned = sample_preprocessed['original_cleaned']
    cleaned_max_jump = pd.Series(cleaned).diff().abs().max()
    assert cleaned_max_jump <= original_max_jump, \
        "IOD correction should reduce maximum jump size"


def test_eph_error_also_processes():
    df = generate_synthetic_gnss_data(seed=42)
    result = preprocess_satellite(df, 'GEO-01', 'EphemerisError_m')
    assert result['error_col'] == 'EphemerisError_m'
    assert not np.any(np.isnan(result['trend']))


def test_meo_also_processes(sample_preprocessed_meo):
    for key in REQUIRED_KEYS:
        assert key in sample_preprocessed_meo
    assert not np.any(np.isnan(sample_preprocessed_meo['trend']))
