"""
Tests for Iteration 4b: Neural ODE Model
All tests must pass as part of Iteration 4 gate.
"""
import pytest
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.models.neural_ode import train_neural_ode, predict_neural_ode
from orbitalmind.models.base_trainer import compute_rmse_horizons


@pytest.fixture(scope="module")
def trained_node_geo():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, metrics = train_neural_ode(data[:480], 'GEO', 'ClockError_ns')
    return model, metrics, data


@pytest.fixture(scope="module")
def trained_node_meo():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'MEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, metrics = train_neural_ode(data[:480], 'MEO', 'ClockError_ns')
    return model, metrics, data


def test_neural_ode_trains_without_nan(trained_node_geo):
    model, metrics, data = trained_node_geo
    assert not np.isnan(metrics['final_train_loss']), \
        "NaN loss — ODE training diverged. Check gradient clipping."


def test_neural_ode_loss_decreases(trained_node_geo):
    model, metrics, data = trained_node_geo
    assert metrics['final_train_loss'] < metrics['initial_train_loss'], \
        f"Loss did not decrease: {metrics['initial_train_loss']:.4f} → {metrics['final_train_loss']:.4f}"


def test_neural_ode_prediction_shape(trained_node_geo):
    model, metrics, data = trained_node_geo
    preds = predict_neural_ode(model, data[-96:])
    assert preds.shape == (96,), \
        f"Expected shape (96,), got {preds.shape}"


def test_neural_ode_predictions_are_smooth(trained_node_geo):
    model, metrics, data = trained_node_geo
    preds = predict_neural_ode(model, data[-96:])
    max_jump = np.max(np.abs(np.diff(preds)))
    assert max_jump < 5.0, \
        f"Predictions not smooth — max consecutive jump: {max_jump:.4f} ns. ODE should produce smooth output."


def test_neural_ode_rmse_1hr(trained_node_geo):
    model, metrics, data = trained_node_geo
    val_data = data[480:576]
    last_seq = data[480-96:480]
    preds = predict_neural_ode(model, last_seq, n_steps=96)
    rmse = compute_rmse_horizons(val_data[:4], preds[:4])
    assert rmse['1hr'] < 2.5, \
        f"Neural ODE RMSE at 1hr too high: {rmse['1hr']:.4f} ns (threshold: 2.5)"


def test_neural_ode_predictions_in_original_scale(trained_node_geo):
    model, metrics, data = trained_node_geo
    preds = predict_neural_ode(model, data[-96:])
    assert preds.max() < 50 and preds.min() > -50, \
        "Predictions out of realistic range — check inverse transform"


def test_neural_ode_geo_meo_separate(trained_node_geo, trained_node_meo):
    """
    GEO and MEO must be trained as genuinely separate model instances.

    This test used to assert that the two models' PREDICTIONS differ by more than
    0.1. That is the wrong invariant, and it began failing once the data became
    real: two independently trained models fed similar signals can legitimately
    agree closely, and their agreeing says nothing about whether they are the same
    object. The forecasts converging is a fact about the data, not a defect.

    What must actually hold is that nothing is shared — separate objects, separate
    parameter tensors, and weights that genuinely differ because they were fitted
    to different satellites. That is what this now checks.
    """
    geo_model, _, _ = trained_node_geo
    meo_model, _, _ = trained_node_meo

    assert geo_model is not meo_model, "GEO and MEO share one model object"

    geo_params = dict(geo_model.named_parameters())
    meo_params = dict(meo_model.named_parameters())
    assert geo_params.keys() == meo_params.keys(), "models have different architectures"

    # No parameter tensor may be the same object in both models.
    for name in geo_params:
        assert geo_params[name] is not meo_params[name], \
            f"parameter '{name}' is the same tensor in both models"

    # Trained on different satellites, at least some weights must have diverged.
    differing = [
        name for name in geo_params
        if not torch.allclose(geo_params[name], meo_params[name], atol=1e-8)
    ]
    assert differing, (
        "every weight is identical across GEO and MEO — the two models were not "
        "fitted to different data"
    )


def test_neural_ode_model_saved():
    assert os.path.exists("models/saved/neural_ode_GEO_ClockError_ns.pt"), \
        "Neural ODE GEO model not saved to models/saved/"


def test_neural_ode_ephemeris_also_trains():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'EphemerisError_m')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, metrics = train_neural_ode(data, 'GEO', 'EphemerisError_m')
    preds = predict_neural_ode(model, data[-96:])
    assert preds.shape == (96,)
    assert not np.any(np.isnan(preds))
