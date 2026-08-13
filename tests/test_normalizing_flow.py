"""
Tests for Iteration 7: Normalizing Flow Post-Processor
This is the hardest gate — Shapiro-Wilk p > 0.05 is the primary eval criterion.
All tests must pass before moving to Iteration 8.
"""
import pytest
import numpy as np
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.models.lstm import train_lstm, predict_lstm
from orbitalmind.models.normalizing_flow import (
    build_normalizing_flow,
    train_normalizing_flow,
    apply_normalizing_flow,
)


def get_ensemble_residuals():
    """Get real residuals from ensemble predictions on validation data."""
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']

    model, _ = train_lstm(data, 'GEO', 'ClockError_ns')
    val_data = data[480:576]
    preds = predict_lstm(model, data[480-96:480], n_steps=96)
    residuals = val_data - preds
    return residuals, val_data, preds


@pytest.fixture(scope="module")
def nf_setup():
    residuals, val_data, preds = get_ensemble_residuals()
    nf_model, res_mean, res_std = train_normalizing_flow(residuals)
    return nf_model, residuals, val_data, preds, res_mean, res_std


def test_nf_builds_without_error():
    model = build_normalizing_flow(input_dim=1, n_flows=4)
    assert model is not None, "Normalizing Flow failed to build"


def test_nf_trains_without_nan(nf_setup):
    nf_model, residuals, val_data, preds, res_mean, res_std = nf_setup
    assert nf_model is not None
    corrected = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    assert not np.any(np.isnan(corrected)), \
        "NaN in NF-corrected predictions"


def test_shapiro_wilk_passes(nf_setup):
    """
    PRIMARY EVALUATION CRITERION.
    Residuals after NF must follow normal distribution.
    p > 0.05 means we cannot reject normality hypothesis.
    """
    nf_model, residuals, val_data, preds, res_mean, res_std = nf_setup
    corrected = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    final_residuals = val_data - corrected
    stat, p_value = stats.shapiro(final_residuals)
    assert p_value > 0.05, \
        f"Shapiro-Wilk FAILED: p={p_value:.4f} (need p > 0.05). " \
        f"Residuals not Gaussian. Increase n_flows or training epochs."


def test_kl_divergence_acceptable(nf_setup):
    nf_model, residuals, val_data, preds, res_mean, res_std = nf_setup
    corrected = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    final_residuals = val_data - corrected
    res_norm = (final_residuals - final_residuals.mean()) / (final_residuals.std() + 1e-8)
    hist, bin_edges = np.histogram(res_norm, bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    gaussian = stats.norm.pdf(bin_centers, 0, 1)
    hist = hist + 1e-10
    gaussian = gaussian + 1e-10
    kl_div = np.sum(hist * np.log(hist / gaussian)) * (bin_edges[1] - bin_edges[0])
    assert kl_div < 0.5, \
        f"KL divergence too high: {kl_div:.4f} (threshold: 0.5). Distribution not Gaussian enough."


def test_nf_corrected_shape(nf_setup):
    nf_model, residuals, val_data, preds, res_mean, res_std = nf_setup
    corrected = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    assert corrected.shape == preds.shape, \
        f"NF output shape {corrected.shape} must match input shape {preds.shape}"


def test_nf_improves_normality_vs_raw(nf_setup):
    nf_model, residuals, val_data, preds, res_mean, res_std = nf_setup
    raw_residuals = val_data - preds
    _, p_raw = stats.shapiro(raw_residuals)
    corrected = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    corrected_residuals = val_data - corrected
    _, p_corrected = stats.shapiro(corrected_residuals)
    assert p_corrected >= p_raw, \
        f"NF made residuals LESS Gaussian: p_raw={p_raw:.4f}, p_corrected={p_corrected:.4f}"


def test_qq_plot_saved(nf_setup):
    nf_model, residuals, val_data, preds, res_mean, res_std = nf_setup
    from orbitalmind.evaluation.gaussian_check import save_qq_plot
    corrected = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    final_residuals = val_data - corrected
    save_qq_plot(final_residuals, path="outputs/qq_plot.png")
    assert os.path.exists("outputs/qq_plot.png"), \
        "Q-Q plot not saved to outputs/qq_plot.png"


def test_nf_model_saved():
    assert os.path.exists("models/saved/normalizing_flow_GEO_ClockError_ns.pt"), \
        "Normalizing Flow model not saved"
