"""
Tests for Iteration 5: Temporal Fusion Transformer
All tests must pass before moving to Iteration 6.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.models.tft import train_tft, predict_tft, prepare_tft_dataframe
from orbitalmind.models.lstm import train_lstm, predict_lstm
from orbitalmind.models.base_trainer import compute_rmse_horizons


@pytest.fixture(scope="module")
def tft_setup():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    tft_df = prepare_tft_dataframe(preprocessed).iloc[:480].reset_index(drop=True)
    model, metrics = train_tft(tft_df, 'GEO', 'ClockError_ns')
    return model, metrics, data, tft_df


@pytest.fixture(scope="module")
def lstm_24hr_rmse():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'GEO-01', 'ClockError_ns')
    data = preprocessed['trend'] + preprocessed['periodic']
    model, _ = train_lstm(data[:480], 'GEO', 'ClockError_ns')
    val_data = data[480:576]
    preds = predict_lstm(model, data[480-96:480], n_steps=96)
    rmse = compute_rmse_horizons(val_data, preds)
    return rmse['24hr']


def test_tft_trains_without_error(tft_setup):
    model, metrics, data, tft_df = tft_setup
    assert model is not None, "TFT model is None — training failed"
    assert 'final_val_loss' in metrics, "No validation loss recorded"


def test_tft_prediction_shape(tft_setup):
    model, metrics, data, tft_df = tft_setup
    preds = predict_tft(model, data[480 - 96:480], n_steps=96)
    assert preds.shape == (96,), \
        f"Expected (96,), got {preds.shape}"


def test_tft_beats_lstm_at_24hr(tft_setup, lstm_24hr_rmse):
    model, metrics, data, tft_df = tft_setup
    val_data = data[480:576]
    preds = predict_tft(model, data[480 - 96:480], n_steps=96)
    rmse = compute_rmse_horizons(val_data, preds)
    assert rmse['24hr'] < lstm_24hr_rmse, \
        f"TFT RMSE at 24hr ({rmse['24hr']:.4f}) must beat LSTM ({lstm_24hr_rmse:.4f})"


def test_tft_prediction_not_nan(tft_setup):
    model, metrics, data, tft_df = tft_setup
    preds = predict_tft(model, data[480 - 96:480], n_steps=96)
    assert not np.any(np.isnan(preds)), \
        "NaN values in TFT predictions"


def test_tft_model_saved():
    assert os.path.exists("models/saved/tft_GEO_ClockError_ns.ckpt"), \
        "TFT checkpoint not saved"


def test_tft_meo_trains_separately():
    df = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, 'MEO-01', 'ClockError_ns')
    tft_df = prepare_tft_dataframe(preprocessed).iloc[:480].reset_index(drop=True)
    model, metrics = train_tft(tft_df, 'MEO', 'ClockError_ns')
    assert model is not None


def test_tft_fallback_flag_on_failure():
    """If TFT fails, a fallback flag must be set — pipeline must not crash."""
    from orbitalmind.models.tft import TFT_FALLBACK_FLAG
    assert isinstance(TFT_FALLBACK_FLAG, bool), \
        "TFT_FALLBACK_FLAG must be a boolean exported from tft.py"


def test_tft_rmse_1hr_acceptable(tft_setup):
    model, metrics, data, tft_df = tft_setup
    val_data = data[480:576]
    preds = predict_tft(model, data[480 - 96:480], n_steps=96)
    rmse = compute_rmse_horizons(val_data[:4], preds[:4])
    assert rmse['1hr'] < 3.0, \
        f"TFT RMSE at 1hr too high: {rmse['1hr']:.4f}. TFT is weaker at short horizons but must be reasonable."
