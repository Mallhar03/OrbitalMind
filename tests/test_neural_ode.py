"""
Tests for Iteration 4b: Neural ODE Model
All tests must pass as part of Iteration 4 gate.
"""
import pytest
import numpy as np
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
    model, metrics = train_neural_ode(data, 'GEO', 'ClockError_ns')
    return model, metrics, data


@pytest.fixture(scope="module")
def trained_node_meo():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'MEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, metrics = train_neural_ode(data, 'MEO', 'ClockError_ns')
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
    geo_model, _, geo_data = trained_node_geo
    meo_model, _, meo_data = trained_node_meo
    geo_preds = predict_neural_ode(geo_model, geo_data[-96:])
    meo_preds = predict_neural_ode(meo_model, meo_data[-96:])
    assert not np.allclose(geo_preds, meo_preds, atol=0.1), \
        "GEO and MEO predictions are identical — models may not be separate instances"


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
