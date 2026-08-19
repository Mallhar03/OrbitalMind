"""
LightGBM stacking meta-learner that fuses LSTM, TCN-LSTM, TFT, and Neural ODE outputs
into a single optimal prediction.
"""
import os
import numpy as np
import lightgbm as lgb

from orbitalmind.paths import models_dir

SAVE_DIR = models_dir()

_LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       15,
    "learning_rate":    0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "seed":             42,
}
_NUM_ROUNDS = 200


def _stack(model_outputs: dict, feature_order: list) -> np.ndarray:
    """Stack model predictions in a consistent feature order."""
    return np.column_stack([
        np.asarray(model_outputs[k], dtype=np.float64) for k in feature_order
    ])


def train_meta_learner(
    model_outputs: dict,
    y_true: np.ndarray,
    orbit_type: str = "GEO",
    error_col: str = "ClockError_ns",
) -> lgb.Booster:
    """
    Train a LightGBM stacking model on base model predictions.

    Trains on all available samples (no val split) to maximise in-sample fit.
    Four base-model predictions are the features; actual values are the target.

    Args:
        model_outputs: dict mapping model name → (n,) prediction array.
                       Required keys: lstm, tcn_lstm, tft, neural_ode.
        y_true: (n,) array of actual target values for the same time range.
        orbit_type: used for the saved model filename (default 'GEO').
        error_col:  used for the saved model filename (default 'ClockError_ns').
    Returns:
        Trained lgb.Booster with feature names stored internally.
    """
    feature_order = list(model_outputs.keys())
    X = _stack(model_outputs, feature_order)
    y = np.asarray(y_true, dtype=np.float64)

    train_set = lgb.Dataset(X, y, feature_name=feature_order)
    model = lgb.train(_LGB_PARAMS, train_set, num_boost_round=_NUM_ROUNDS)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_model(f"{SAVE_DIR}/meta_learner_{orbit_type}_{error_col}.txt")

    return model


def predict_meta_learner(
    model: lgb.Booster,
    model_outputs: dict,
) -> np.ndarray:
    """
    Generate ensemble predictions using the trained meta-learner.

    Args:
        model: trained lgb.Booster from train_meta_learner
        model_outputs: dict mapping model name → (n,) prediction array
    Returns:
        np.ndarray of shape (n,) — fused predictions.
    """
    feature_order = model.feature_name()
    X = _stack(model_outputs, feature_order)
    return model.predict(X)


def get_feature_importance(model: lgb.Booster, feature_names: list) -> dict:
    """
    Return the gain-based feature importance for each base model.

    Args:
        model: trained lgb.Booster
        feature_names: list of model names in the same order as training features
    Returns:
        Dict mapping model name → importance score (float).
    """
    importance = model.feature_importance(importance_type="gain")
    model_names = model.feature_name()
    imp_map = dict(zip(model_names, importance.tolist()))
    return {name: imp_map.get(name, 0.0) for name in feature_names}
