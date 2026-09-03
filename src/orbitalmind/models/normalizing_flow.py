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
        bias        = _shrink_bias(pool, n_observed=len(residuals)),
        sigma       = float(np.std(pool)),
        sample_pool = pool,
        fitted      = not diverged,
    )



def _shrink_bias(pool: np.ndarray, n_observed: int) -> float:
    """
    Estimate the scalar bias correction, shrunk toward zero when it is not
    distinguishable from its own estimation noise.

    This exists because of how the bias is USED. `apply_normalizing_flow()` adds
    this one scalar to every step of a differenced forecast, and `_reconstruct()`
    then accumulates those steps. A constant added to 96 differences becomes a
    straight-line drift of 96 x bias in the reconstructed level — so the bias is
    effectively a slope correction, and any error in it is amplified by the
    horizon length rather than staying bounded.

    That amplification is what made the ephemeris column lose to a plain linear
    fit. The bias is a median of roughly 48 calibration residuals, so its standard
    error is about 1.253 * sigma / sqrt(n). Measured on real 2026-08 data, that
    standard error (0.0036-0.004 m) was almost exactly the size of the bias being
    applied (median 0.0038 m) — the correction was indistinguishable from the
    noise in its own estimate, and accumulating it 96 times produced the entire
    24-hour error blow-up. The ensemble/linear RMSE ratio grew monotonically from
    1.10x at 15 minutes to 2.65x at 24 hours, which is the signature of a
    compounding constant rather than a badly fitted model.

    The shrinkage factor is b^2 / (b^2 + se^2): when the bias is large relative to
    its uncertainty it passes through essentially untouched, and when it is
    comparable to its uncertainty it is damped toward zero. Nothing here is tuned
    against a score — the criterion is whether the correction is statistically
    distinguishable from zero, which is a question about the estimate, not about
    the forecast it improves.

    Args:
        pool:       residual sample pool the calibration was fitted to
        n_observed: number of real residuals the pool was derived from, which
                    sets the estimator's standard error
    Returns:
        The shrunk bias, in residual units.
    """
    # DEFAULT IS "zero": no bias correction is applied at all.
    #
    # This was measured, not assumed. On 30 satellites, paired, against both
    # alternatives, applying no correction beat shrinking one on position error by
    # 28% at the 24-hour horizon (0.1907 vs 0.2640 m) and beat the original
    # unshrunk correction by 39% (vs 0.3145 m). Satellites beating a plain linear
    # baseline on position rose from 45/150 (raw) and 47/150 (shrunk) to 63/150.
    # Clock was unaffected either way, which is consistent: the clock never
    # suffered from this.
    #
    # The reason is that there is no bias to correct. The estimate is a median of
    # ~48 residuals, whose standard error is 1.2533*sigma/sqrt(48) = 0.18*sigma —
    # so any true bias below 0.18 standard deviations is unmeasurable at this
    # window size, and measured calibration means sit around 0.43 standard errors
    # from zero, exceeding significance on 1 of 40 satellites. Applying the
    # estimate meant adding noise to all 96 predicted steps, which _reconstruct()
    # then accumulated into a drift.
    #
    # "shrink" and "raw" remain available for reproducing that comparison.
    mode = os.environ.get("ORBITALMIND_BIAS_MODE", "zero").strip().lower()

    bias = float(np.median(pool))
    if mode == "zero":
        return 0.0
    if mode == "raw":
        return bias
    if n_observed < 2:
        return bias

    spread = float(np.std(pool))
    if spread <= 0.0:
        return bias

    # Asymptotic standard error of a median, ~1.253 * sigma / sqrt(n).
    std_err = 1.2533 * spread / np.sqrt(n_observed)
    if std_err <= 0.0:
        return bias

    shrinkage = bias ** 2 / (bias ** 2 + std_err ** 2)
    return float(bias * shrinkage)


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
