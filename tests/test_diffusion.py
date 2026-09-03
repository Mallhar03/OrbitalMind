"""
Tests for Iteration 4c: Diffusion Model for Residuals
All tests must pass as part of Iteration 4 gate.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.models.lstm import train_lstm, predict_lstm
from orbitalmind.models.diffusion import train_diffusion, sample_residuals


def compute_residuals_array(n_windows=20):
    """Build a real residuals array from LSTM predictions."""
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, _ = train_lstm(data, 'GEO', 'ClockError_ns')

    residuals = []
    for i in range(n_windows):
        start = i * 24
        if start + 96 + 96 > len(data):
            break
        seq = data[start:start+96]
        actual = data[start+96:start+192]
        preds = predict_lstm(model, seq, n_steps=96)
        residuals.append(actual - preds)

    return np.array(residuals)


@pytest.fixture(scope="module")
def trained_diffusion():
    residuals = compute_residuals_array()
    model = train_diffusion(residuals)
    return model, residuals


def test_diffusion_trains_without_error(trained_diffusion):
    model, residuals = trained_diffusion
    assert model is not None, "Diffusion model is None"


def test_diffusion_sample_shape(trained_diffusion):
    model, residuals = trained_diffusion
    samples = sample_residuals(model, n_samples=1, seq_len=96)
    assert samples.shape == (1, 96), \
        f"Expected shape (1, 96), got {samples.shape}"


def test_diffusion_sample_mean_close_to_training(trained_diffusion):
    model, residuals = trained_diffusion
    samples = sample_residuals(model, n_samples=50, seq_len=96)
    train_mean = residuals.mean()
    sample_mean = samples.mean()
    train_std = residuals.std()
    assert abs(sample_mean - train_mean) < 2 * train_std, \
        f"Sample mean {sample_mean:.4f} too far from training mean {train_mean:.4f}"


def test_diffusion_sample_std_reasonable(trained_diffusion):
    model, residuals = trained_diffusion
    samples = sample_residuals(model, n_samples=50, seq_len=96)
    train_std = residuals.std()
    sample_std = samples.std()
    assert 0.5 * train_std < sample_std < 2.0 * train_std, \
        f"Sample std {sample_std:.4f} not within 50% of training std {train_std:.4f}"


def test_diffusion_samples_not_identical(trained_diffusion):
    model, residuals = trained_diffusion
    s1 = sample_residuals(model, n_samples=1, seq_len=96)
    s2 = sample_residuals(model, n_samples=1, seq_len=96)
    assert not np.allclose(s1, s2), \
        "Diffusion samples are identical — model is not stochastic"


def test_diffusion_no_nan_in_samples(trained_diffusion):
    model, residuals = trained_diffusion
    samples = sample_residuals(model, n_samples=10, seq_len=96)
    assert not np.any(np.isnan(samples)), \
        "NaN in diffusion samples — check reverse diffusion process"


def test_diffusion_input_shape_validated():
    residuals_wrong_shape = np.random.randn(20)
    with pytest.raises((ValueError, AssertionError)):
        train_diffusion(residuals_wrong_shape)


def test_diffusion_ephemeris_residuals_also_work():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'EphemerisError_m')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, _ = train_lstm(data, 'GEO', 'EphemerisError_m')
    residuals = []
    for i in range(10):
        start = i * 24
        if start + 192 > len(data):
            break
        seq = data[start:start+96]
        actual = data[start+96:start+192]
        preds = predict_lstm(model, seq, n_steps=96)
        residuals.append(actual - preds)
    residuals = np.array(residuals)
    diff_model = train_diffusion(residuals)
    samples = sample_residuals(diff_model, n_samples=1, seq_len=96)
    assert samples.shape == (1, 96)
