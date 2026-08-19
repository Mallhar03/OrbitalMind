"""
Tests for Iteration 4: LSTM and TCN-LSTM Models
All tests must pass before moving to Iteration 5.
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
from orbitalmind.models.base_trainer import compute_rmse_horizons


@pytest.fixture(scope="module")
def trained_lstm():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    # train_lstm fits exactly what it is handed; the caller owns the window,
    # so data[480:576] below stays genuinely out-of-sample.
    model, metrics = train_lstm(data[:480], 'GEO', 'ClockError_ns')
    return model, metrics, data


def test_lstm_trains(trained_lstm):
    model, metrics, data = trained_lstm
    assert model is not None
    assert 'final_train_loss' in metrics
    assert metrics['final_train_loss'] < metrics.get('initial_train_loss', 999)


def test_lstm_prediction_shape(trained_lstm):
    model, metrics, data = trained_lstm
    last_seq = data[-96:]
    preds = predict_lstm(model, last_seq)
    assert preds.shape == (96,), f"Expected (96,), got {preds.shape}"


def test_lstm_rmse_15min(trained_lstm):
    model, metrics, data = trained_lstm
    train_data = data[:480]
    val_data   = data[480:576]
    last_seq   = train_data[-96:]
    preds      = predict_lstm(model, last_seq, n_steps=96)
    rmse = compute_rmse_horizons(val_data[:1], preds[:1])
    assert rmse['15min'] < 1.5, f"RMSE at 15min too high: {rmse['15min']:.4f}"


def test_lstm_rmse_1hr(trained_lstm):
    model, metrics, data = trained_lstm
    val_data = data[480:576]
    last_seq = data[480-96:480]
    preds    = predict_lstm(model, last_seq, n_steps=96)
    rmse     = compute_rmse_horizons(val_data[:4], preds[:4])
    assert rmse['1hr'] < 2.0, f"RMSE at 1hr too high: {rmse['1hr']:.4f}"


def test_lstm_model_saved():
    assert os.path.exists("models/saved/lstm_GEO_ClockError_ns.pt"), \
        "LSTM model not saved"


def test_tcn_lstm_prediction_shape():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'MEO-01', 'ClockError_ns')
    data  = preprocessed['trend'] + preprocessed['periodic']
    model, _ = train_tcn_lstm(data, 'MEO', 'ClockError_ns')
    preds = predict_tcn_lstm(model, data[-96:])
    assert preds.shape == (96,), f"TCN-LSTM prediction shape wrong: {preds.shape}"


def test_predictions_in_original_scale(trained_lstm):
    model, metrics, data = trained_lstm
    preds = predict_lstm(model, data[-96:])
    assert preds.max() < 50, "Predictions seem unnormalised — check inverse transform"
    assert preds.min() > -50, "Predictions seem unnormalised — check inverse transform"
