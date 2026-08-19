"""
Normalizing Flow residual calibrator for the GNSS ensemble.

The flow is fitted to out-of-sample residuals (actual - predicted) from a
calibration window that the base models never trained on. It then supplies
two things the proposal promises for every forecast point:

  * a scalar bias correction for the point estimate, and
  * a predictive distribution (sigma and quantile bounds) around it.

What this deliberately does NOT do
----------------------------------
The previous version computed

    _correction = (z_gaussian * res_std + res_mean) - residuals
    corrected   = preds - _correction

from the very ground truth it was later scored against, which made
`actual - corrected` a vector of exact normal quantiles by construction.
Shapiro-Wilk then returned p ~= 0.9999 for any input whatsoever, including
residuals that were half constant and half exponential. It measured nothing.
Worse, that same ground-truth-derived vector was subtracted from the day-8
forecast, injecting a different day's errors into the prediction.

Normality is now something the pipeline *reports* on held-out residuals. If
those residuals are not Gaussian, the test says so.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import normflows as nf
from scipy import stats

from orbitalmind.paths import models_dir

EPOCHS    = 200
LR        = 0.001
GRAD_CLIP = 1.0
N_SAMPLES = 4000          # flow samples drawn to estimate spread and quantiles
SAVE_DIR = models_dir()


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


@dataclass
class ResidualCalibration:
    """
    Learned residual distribution for one satellite/error-column combination.

    Every field is a summary statistic of the calibration residuals. None of
    them is (or is derived index-by-index from) the targets the forecast is
    later scored against, which is the property test_normalizing_flow.py pins
    down.

    Attributes:
        bias:           scalar shift removed from point predictions
        sigma:          standard deviation of the learned residual law
        sample_pool:    residual draws from the flow, used for quantiles
        fitted:         False if the flow diverged and Gaussian fallback was used
    """
    bias:        float
    sigma:       float
    sample_pool: np.ndarray = field(repr=False)
    fitted:      bool = True

    def quantile(self, q: float) -> float:
        """Return the q-th quantile of the learned residual law (q in [0, 1])."""
        return float(np.quantile(self.sample_pool, q))


def _fallback_pool(residuals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian sample pool used when the flow fails to converge."""
    return rng.normal(float(np.mean(residuals)),
                      float(np.std(residuals)) + 1e-12, N_SAMPLES)


def train_normalizing_flow(
    residuals:  np.ndarray,
    epochs:     int = EPOCHS,
    orbit_type: str = "GEO",
    error_col:  str = "ClockError_ns",
) -> ResidualCalibration:
    """
    Fit a normalizing flow to out-of-sample ensemble residuals.

    The residuals are standardised, the flow is fitted by maximum likelihood,
    and a pool of samples is drawn from it and mapped back to residual units.
    Bias and spread are read off that pool. If the flow diverges, the pool
    falls back to a Gaussian with the empirical mean and standard deviation.

    Args:
        residuals:  1-D array of (actual - predicted) on the calibration window
        epochs:     maximum training epochs
        orbit_type: 'GEO' or 'MEO' — used for the checkpoint filename
        error_col:  'ClockError_ns' or 'EphemerisError_m' — checkpoint filename
    Returns:
        ResidualCalibration describing the learned residual law.
    """
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    residuals = np.asarray(residuals, dtype=np.float64).flatten()
    res_mean  = float(np.mean(residuals))
    res_std   = float(np.std(residuals))

    # Degenerate calibration window (constant residuals): no spread to learn.
    if res_std < 1e-12:
        pool = np.full(N_SAMPLES, res_mean)
        return ResidualCalibration(bias=res_mean, sigma=0.0,
                                   sample_pool=pool, fitted=False)

    r_norm = np.clip((residuals - res_mean) / res_std, -5.0, 5.0)

    model     = build_normalizing_flow()
    X         = torch.tensor(r_norm.reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    diverged = False
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = -model.log_prob(X).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            diverged = True
            break
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    if diverged:
        pool = _fallback_pool(residuals, rng)
    else:
        model.eval()
        with torch.no_grad():
            z, _ = model.sample(N_SAMPLES)
            drawn = z.numpy().flatten()
        if not np.all(np.isfinite(drawn)):
            pool = _fallback_pool(residuals, rng)
            diverged = True
        else:
            # Back to residual units.
            pool = drawn * res_std + res_mean

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"{SAVE_DIR}/normalizing_flow_{orbit_type}_{error_col}.pt",
    )

    return ResidualCalibration(
        bias        = float(np.median(pool)),
        sigma       = float(np.std(pool)),
        sample_pool = pool,
        fitted      = not diverged,
    )


def apply_normalizing_flow(
    calibration: ResidualCalibration,
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Bias-correct point predictions using the learned residual law.

    Residuals are defined as (actual - predicted), so a positive learned bias
    means the ensemble runs low and the correction adds it back. The shift is
    a single scalar: it cannot encode anything index-specific about the window
    being predicted.

    Args:
        calibration: result of train_normalizing_flow()
        predictions: (n,) base ensemble predictions
    Returns:
        (n,) bias-corrected predictions as float64.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    return preds + calibration.bias


def predictive_interval(
    calibration: ResidualCalibration,
    predictions: np.ndarray,
    level:       float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the predictive distribution around each forecast point.

    This is the "probability distribution from Normalizing Flow" half of the
    proposal's two-outputs-per-point mechanism. Bounds come from quantiles of
    the fitted residual law, so an asymmetric residual distribution yields an
    asymmetric interval.

    Args:
        calibration: result of train_normalizing_flow()
        predictions: (n,) point predictions (already bias-corrected)
        level:       nominal coverage, e.g. 0.95
    Returns:
        (lower, upper, sigma), each (n,) float64 arrays.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    alpha = (1.0 - level) / 2.0

    centred = calibration.sample_pool - calibration.bias
    lo = float(np.quantile(centred, alpha))
    hi = float(np.quantile(centred, 1.0 - alpha))

    sigma = np.full(preds.shape, max(calibration.sigma, 1e-12), dtype=np.float64)
    return preds + lo, preds + hi, sigma


def shapiro_wilk(residuals: np.ndarray) -> tuple[float, float, str]:
    """
    Report the Shapiro-Wilk normality test on residuals as they actually are.

    Args:
        residuals: 1-D array of held-out (actual - predicted) values
    Returns:
        (statistic, p_value, 'PASS' if p > 0.05 else 'FAIL').
    """
    res = np.asarray(residuals, dtype=np.float64).flatten()
    res = res[np.isfinite(res)][:5000]
    if len(res) < 3 or np.std(res) < 1e-15:
        return 0.0, 0.0, "FAIL"
    stat, p = stats.shapiro(res)
    return float(stat), float(p), ("PASS" if p > 0.05 else "FAIL")
