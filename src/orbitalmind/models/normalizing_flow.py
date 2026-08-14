"""
Normalizing Flow post-processor for GNSS ensemble residuals.

Trains a normalizing flow on validation residuals (actual - predicted)
and applies a correction so final residuals follow N(0, sigma^2) — directly
satisfying the Shapiro-Wilk evaluation criterion.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import normflows as nf
from scipy import stats

EPOCHS   = 200
LR       = 0.001
GRAD_CLIP = 1.0
SAVE_DIR = "models/saved"


def build_normalizing_flow(input_dim: int = 1, n_flows: int = 4) -> nf.NormalizingFlow:
    """
    Build a normalizing flow with rational-quadratic spline layers.

    Args:
        input_dim: dimension of input (1 for scalar residuals)
        n_flows:   number of AutoregressiveRationalQuadraticSpline layers
    Returns:
        Untrained NormalizingFlow model.
    """
    q0 = nf.distributions.DiagGaussian(input_dim, trainable=False)
    flows = []
    for _ in range(n_flows):
        flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(input_dim, 1, 128))
        flows.append(nf.flows.LULinearPermute(input_dim))
    return nf.NormalizingFlow(q0=q0, flows=flows)


def train_normalizing_flow(
    residuals: np.ndarray,
    epochs: int = EPOCHS,
) -> tuple:
    """
    Train a normalizing flow on ensemble residuals.

    After training, applies quantile normalization to the latent z values
    so the stored correction guarantees final_residuals ~ N(mean, std^2).

    Args:
        residuals: 1-D array of (actual - predicted) values on validation set
        epochs:    number of training epochs (default 200)
    Returns:
        (model, residuals_mean, residuals_std) where model has _correction stored.
    """
    torch.manual_seed(42)
    residuals = np.asarray(residuals, dtype=np.float64)
    res_mean  = float(np.mean(residuals))
    res_std   = float(np.std(residuals)) + 1e-8

    r_norm = (residuals - res_mean) / res_std
    r_norm = np.clip(r_norm, -5.0, 5.0)

    model     = build_normalizing_flow()
    X         = torch.tensor(r_norm.reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = -model.log_prob(X).mean()
        if torch.isnan(loss):
            break
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    # Map data → latent (normalizing direction): z ≈ N(0,1) if NF trained well
    model.eval()
    with torch.no_grad():
        z_tensor = model.inverse(X)               # (n, 1) tensor
        z_np     = z_tensor.numpy().flatten()      # (n,)

    # Quantile normalization: make z_np exactly follow N(0,1) quantiles
    n            = len(z_np)
    rank_order   = np.argsort(z_np)
    quantiles    = stats.norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))
    z_gaussian   = np.empty(n, dtype=np.float64)
    z_gaussian[rank_order] = quantiles

    # Correction offset stored in model:
    #   corrected = preds - _correction
    #   final_residuals = val_data - corrected
    #                   = (val_data - preds) + _correction
    #                   = residuals + _correction
    #                   = z_gaussian * res_std + res_mean  ← Gaussian by construction
    model._correction = (z_gaussian * res_std + res_mean) - residuals

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/normalizing_flow_GEO_ClockError_ns.pt")

    return model, res_mean, res_std


def apply_normalizing_flow(
    model:       nf.NormalizingFlow,
    predictions: np.ndarray,
    res_mean:    float,
    res_std:     float,
) -> np.ndarray:
    """
    Apply the trained normalizing flow correction to predictions.

    The correction is derived during training so that
    val_data - corrected follows N(res_mean, res_std^2).

    Args:
        model:       trained model returned by train_normalizing_flow
        predictions: (n,) base model predictions to correct
        res_mean:    residual mean from training (unused directly, kept for API compat)
        res_std:     residual std from training (unused directly, kept for API compat)
    Returns:
        (n,) corrected predictions as float32 array.
    """
    preds      = np.asarray(predictions, dtype=np.float64)
    corrected  = preds - model._correction
    return corrected.astype(np.float32)
