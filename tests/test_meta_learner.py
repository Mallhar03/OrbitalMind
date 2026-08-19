"""
Tests for Iteration 6: LightGBM Meta-Learner
All tests must pass before moving to Iteration 7.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.models.lstm import train_lstm, predict_lstm
from orbitalmind.models.tcn_lstm import train_tcn_lstm, predict_tcn_lstm
from orbitalmind.models.tft import train_tft, predict_tft, prepare_tft_dataframe
from orbitalmind.models.neural_ode import train_neural_ode, predict_neural_ode
from orbitalmind.ensemble.lightgbm_meta import (
    train_meta_learner, predict_meta_learner, get_feature_importance
)
from orbitalmind.models.base_trainer import compute_rmse_horizons


@pytest.fixture(scope="module")
def all_model_outputs():
    """Get predictions from all 4 base models on validation data."""
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']

    train_data = data[:480]
    val_data = data[480:576]
    last_train_seq = train_data[-96:]

    lstm_model, _ = train_lstm(train_data, 'GEO', 'ClockError_ns')
    tcn_model, _ = train_tcn_lstm(train_data, 'GEO', 'ClockError_ns')
    node_model, _ = train_neural_ode(train_data, 'GEO', 'ClockError_ns')

    tft_df = prepare_tft_dataframe(preprocessed).iloc[:480].reset_index(drop=True)
    tft_model, _ = train_tft(tft_df, 'GEO', 'ClockError_ns')

    outputs = {
        'lstm':       predict_lstm(lstm_model, last_train_seq, n_steps=96),
        'tcn_lstm':   predict_tcn_lstm(tcn_model, last_train_seq, n_steps=96),
        'neural_ode': predict_neural_ode(node_model, last_train_seq, n_steps=96),
        'tft':        predict_tft(tft_model, last_train_seq, n_steps=96),
    }
    return outputs, val_data, data


@pytest.fixture(scope="module")
def trained_meta(all_model_outputs):
    outputs, val_data, data = all_model_outputs
    model = train_meta_learner(outputs, val_data)
    return model, outputs, val_data


def test_meta_learner_trains(trained_meta):
    model, outputs, val_data = trained_meta
    assert model is not None, "Meta-learner model is None"


def test_meta_learner_has_all_4_inputs(all_model_outputs):
    outputs, val_data, data = all_model_outputs
    required_keys = {'lstm', 'tcn_lstm', 'neural_ode', 'tft'}
    assert set(outputs.keys()) == required_keys, \
        f"Missing model outputs. Got: {set(outputs.keys())}"


def test_meta_prediction_shape(trained_meta):
    model, outputs, val_data = trained_meta
    preds = predict_meta_learner(model, outputs)
    assert preds.shape == (96,), \
        f"Expected (96,), got {preds.shape}"


def test_meta_beats_best_individual_1hr(trained_meta, all_model_outputs):
    model, outputs, val_data = trained_meta
    meta_preds = predict_meta_learner(model, outputs)
    meta_rmse = compute_rmse_horizons(val_data[:4], meta_preds[:4])['1hr']

    individual_rmses = {
        name: compute_rmse_horizons(val_data[:4], preds[:4])['1hr']
        for name, preds in outputs.items()
    }
    best_individual = min(individual_rmses.values())

    assert meta_rmse < best_individual, \
        f"Meta-learner RMSE {meta_rmse:.4f} must beat best individual {best_individual:.4f}"


def test_meta_beats_best_individual_24hr(trained_meta, all_model_outputs):
    model, outputs, val_data = trained_meta
    meta_preds = predict_meta_learner(model, outputs)
    meta_rmse = compute_rmse_horizons(val_data, meta_preds)['24hr']

    individual_rmses = {
        name: compute_rmse_horizons(val_data, preds)['24hr']
        for name, preds in outputs.items()
    }
    best_individual = min(individual_rmses.values())

    assert meta_rmse < best_individual, \
        f"Meta-learner 24hr RMSE {meta_rmse:.4f} must beat best individual {best_individual:.4f}"


def test_feature_importance_all_models_contribute(trained_meta):
    model, outputs, val_data = trained_meta
    importance = get_feature_importance(model, list(outputs.keys()))
    for name, imp in importance.items():
        assert imp > 0, \
            f"Model {name} has zero importance — it is not contributing to ensemble"


def test_meta_model_saved():
    assert os.path.exists("models/saved/meta_learner_GEO_ClockError_ns.txt"), \
        "Meta-learner model not saved"


def test_zeros_placeholder_does_not_crash(trained_meta):
    """If a model fails, zeros placeholder must not crash meta-learner."""
    model, outputs, val_data = trained_meta
    outputs_with_zero = dict(outputs)
    outputs_with_zero['tft'] = np.zeros(96)
    preds = predict_meta_learner(model, outputs_with_zero)
    assert preds.shape == (96,)
    assert not np.any(np.isnan(preds))
