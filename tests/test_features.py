"""
Tests for Iteration 3: Feature Engineering
All tests must pass before moving to Iteration 4.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.features.feature_matrix import build_feature_matrix

EXPECTED_N_FEATURES = 29


@pytest.fixture(scope="module")
def feature_data_geo():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    sat_df = df[df['SatelliteID'] == 'GEO-01'].sort_values('Timestamp')
    timestamps = pd.to_datetime(sat_df['Timestamp'])
    X, y, names = build_feature_matrix(preprocessed, timestamps)
    return X, y, names


@pytest.fixture(scope="module")
def feature_data_meo():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'MEO-01', 'ClockError_ns')
    sat_df = df[df['SatelliteID'] == 'MEO-01'].sort_values('Timestamp')
    timestamps = pd.to_datetime(sat_df['Timestamp'])
    X, y, names = build_feature_matrix(preprocessed, timestamps)
    return X, y, names


def test_feature_count(feature_data_geo):
    X, y, names = feature_data_geo
    assert X.shape[1] == EXPECTED_N_FEATURES, \
        f"Expected {EXPECTED_N_FEATURES} features, got {X.shape[1]}"


def test_no_nan_in_X(feature_data_geo):
    X, y, names = feature_data_geo
    assert not np.any(np.isnan(X)), "NaN in feature matrix X"


def test_no_nan_in_y(feature_data_geo):
    X, y, names = feature_data_geo
    assert not np.any(np.isnan(y)), "NaN in target y"


def test_normalisation_range(feature_data_geo):
    X, y, names = feature_data_geo
    assert X.min() >= -0.01 and X.max() <= 1.01, \
        f"Features not normalised to [0,1]. Range: [{X.min():.3f}, {X.max():.3f}]"


def test_feature_names_match_columns(feature_data_geo):
    X, y, names = feature_data_geo
    assert len(names) == X.shape[1], \
        "Feature names count does not match column count"


def test_geo_stronger_24hr_signal(feature_data_geo):
    X, y, names = feature_data_geo
    amp_12hr_idx = names.index('amplitude_12hr') if 'amplitude_12hr' in names else None
    amp_24hr_idx = names.index('amplitude_24hr') if 'amplitude_24hr' in names else None
    if amp_12hr_idx is not None and amp_24hr_idx is not None:
        assert X[:, amp_24hr_idx].mean() > X[:, amp_12hr_idx].mean(), \
            "GEO should have stronger 24hr signal than 12hr"


def test_meo_stronger_12hr_signal(feature_data_meo):
    X, y, names = feature_data_meo
    amp_12hr_idx = names.index('amplitude_12hr') if 'amplitude_12hr' in names else None
    amp_24hr_idx = names.index('amplitude_24hr') if 'amplitude_24hr' in names else None
    if amp_12hr_idx is not None and amp_24hr_idx is not None:
        assert X[:, amp_12hr_idx].mean() > X[:, amp_24hr_idx].mean(), \
            "MEO should have stronger 12hr signal than 24hr"
