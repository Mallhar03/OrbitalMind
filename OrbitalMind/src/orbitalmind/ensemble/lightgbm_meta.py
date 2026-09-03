"""
Linear stacking meta-learner that fuses LSTM, TCN-LSTM, TFT, and Neural ODE outputs
into a single optimal smooth prediction (avoids decision tree step artifacts).
"""
import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge

from orbitalmind.paths import models_dir

SAVE_DIR = models_dir()


class RidgeWrapper:
    """Wrapper to make Ridge behave like lgb.Booster for our pipeline."""
    def __init__(self, model: Ridge, feature_name: list):
        self.model = model
        self._feature_name = feature_name

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_name(self) -> list:
        return self._feature_name

    def feature_importance(self, importance_type="gain") -> np.ndarray:
        return np.abs(self.model.coef_)


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
) -> RidgeWrapper:
    """
    Train a Ridge stacking model on base model predictions.

    Trains on all available samples (no val split) to maximise in-sample fit.
    Four base-model predictions are the features; actual values are the target.

    Args:
        model_outputs: dict mapping model name → (n,) prediction array.
        y_true: (n,) array of actual target values.
        orbit_type: used for the saved model filename.
        error_col:  used for the saved model filename.
    Returns:
        Trained RidgeWrapper.
    """
    feature_order = list(model_outputs.keys())
    X = _stack(model_outputs, feature_order)
    y = np.asarray(y_true, dtype=np.float64)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    os.makedirs(SAVE_DIR, exist_ok=True)
    wrapper = RidgeWrapper(model, feature_order)
    with open(f"{SAVE_DIR}/meta_learner_{orbit_type}_{error_col}.pkl", "wb") as f:
        pickle.dump(wrapper, f)

    return wrapper


def predict_meta_learner(
    model: RidgeWrapper,
    model_outputs: dict,
) -> np.ndarray:
    """
    Generate ensemble predictions using the trained meta-learner.

    Args:
        model: trained RidgeWrapper from train_meta_learner
        model_outputs: dict mapping model name → (n,) prediction array
    Returns:
        np.ndarray of shape (n,) — fused predictions.
    """
    feature_order = model.feature_name()
    X = _stack(model_outputs, feature_order)
    return model.predict(X)


def get_feature_importance(model: RidgeWrapper, feature_names: list) -> dict:
    """
    Return the absolute coefficient-based feature importance for each base model.

    Args:
        model: trained RidgeWrapper
        feature_names: list of model names in the same order as training features
    Returns:
        Dict mapping model name → importance score (float).
    """
    importance = model.feature_importance()
    model_names = model.feature_name()
    imp_map = dict(zip(model_names, importance.tolist()))
    return {name: imp_map.get(name, 0.0) for name in feature_names}
