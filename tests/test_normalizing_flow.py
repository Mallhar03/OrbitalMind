"""
Tests for the Normalizing Flow residual calibrator.

The previous implementation derived its correction vector from the validation
ground truth:

    _correction = (z_gaussian * res_std + res_mean) - residuals
    corrected   = preds - _correction

so `val_data - corrected` was *defined* to be a vector of exact normal
quantiles. Shapiro-Wilk therefore returned p ~= 0.9999 for every possible
input, including maximally non-Gaussian residuals. It measured nothing, and
the same ground-truth-derived vector was then subtracted from the day-8
forecast, which actively degraded it.

These tests pin down the properties that make the calibrator honest:
  * the correction is a shift learned from calibration data, never a
    per-index vector keyed to the targets it is scored against;
  * normality is reported, not manufactured;
  * the flow supplies a predictive distribution, not just a point estimate.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scipy import stats
from orbitalmind.models.normalizing_flow import (
    train_normalizing_flow,
    apply_normalizing_flow,
    predictive_interval,
)

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def gaussian_calibration():
    """Calibrator fitted on genuinely Gaussian residuals."""
    residuals = RNG.normal(0.3, 1.2, 96)
    return train_normalizing_flow(residuals), residuals


@pytest.fixture(scope="module")
def skewed_calibration():
    """Calibrator fitted on strongly non-Gaussian residuals."""
    residuals = np.concatenate([np.ones(48) * 50.0, RNG.exponential(3.0, 48)])
    return train_normalizing_flow(residuals), residuals


def test_correction_is_a_shift_not_a_per_index_vector(gaussian_calibration):
    """
    The old bug in one assertion: the correction must depend only on what the
    calibrator learned, never on the index it lands at. Two different
    prediction vectors must therefore be moved by the identical amount.
    """
    cal, _ = gaussian_calibration
    a = RNG.normal(0, 5, 96)
    b = RNG.normal(100, 20, 96)
    shift_a = apply_normalizing_flow(cal, a) - a
    shift_b = apply_normalizing_flow(cal, b) - b
    assert np.allclose(shift_a, shift_b), \
        "correction varies by index — it is keyed to ground truth again"


def test_correction_magnitude_matches_calibration_bias(gaussian_calibration):
    """The learned shift should remove the calibration residual bias."""
    cal, residuals = gaussian_calibration
    preds = RNG.normal(0, 5, 96)
    shift = float(np.mean(apply_normalizing_flow(cal, preds) - preds))
    assert shift == pytest.approx(np.median(residuals), abs=0.5), \
        f"shift {shift:.3f} does not track calibration bias"


def test_non_gaussian_residuals_are_not_laundered(skewed_calibration):
    """
    Feeding maximally non-Gaussian residuals must NOT yield a near-perfect
    normality score. The old code returned p = 0.9999 here.
    """
    cal, residuals = skewed_calibration
    preds = np.zeros(96)
    corrected = apply_normalizing_flow(cal, preds)
    final = (-residuals) - corrected  # residuals that remain after correction
    p = float(stats.shapiro(final)[1])
    assert p < 0.9, \
        f"non-Gaussian residuals reported as Gaussian (p={p:.4f}) — circular again"


def test_calibrator_does_not_see_the_scored_targets(skewed_calibration):
    """
    train_normalizing_flow takes residuals only. It must expose no attribute
    holding a full-length copy of them, which is how the leak happened before.
    """
    cal, residuals = skewed_calibration
    for name in vars(cal):
        value = getattr(cal, name)
        if isinstance(value, np.ndarray) and value.shape == residuals.shape:
            assert not np.allclose(value, residuals), \
                f"attribute {name!r} stores the calibration targets verbatim"


def test_predictive_interval_brackets_the_point_estimate(gaussian_calibration):
    """The distribution output the proposal promises: lower <= point <= upper."""
    cal, _ = gaussian_calibration
    preds = RNG.normal(0, 5, 96)
    lower, upper, sigma = predictive_interval(cal, preds, level=0.95)
    assert lower.shape == upper.shape == sigma.shape == preds.shape
    assert np.all(lower <= upper), "interval inverted"
    assert np.all(sigma > 0), "non-positive predictive sigma"


def test_interval_width_reflects_residual_spread():
    """A noisier calibration set must yield a wider predictive interval."""
    tight = train_normalizing_flow(RNG.normal(0, 0.1, 96))
    loose = train_normalizing_flow(RNG.normal(0, 10.0, 96))
    preds = np.zeros(96)
    tl, tu, _ = predictive_interval(tight, preds, level=0.95)
    ll, lu, _ = predictive_interval(loose, preds, level=0.95)
    assert np.mean(lu - ll) > np.mean(tu - tl), \
        "interval width ignores residual spread"


def test_nominal_coverage_is_approximately_honest():
    """
    A 95% interval calibrated on Gaussian residuals should cover roughly 95%
    of fresh residuals from the same distribution — not 100%, not 50%.
    """
    residuals = RNG.normal(0.0, 2.0, 96)
    cal = train_normalizing_flow(residuals)
    fresh = RNG.normal(0.0, 2.0, 500)
    preds = np.zeros(500)
    lower, upper, _ = predictive_interval(cal, preds, level=0.95)
    covered = np.mean((fresh >= lower) & (fresh <= upper))
    assert 0.80 <= covered <= 1.0, f"coverage {covered:.2f} far from nominal 0.95"


def test_no_nan_produced(skewed_calibration):
    cal, _ = skewed_calibration
    preds = RNG.normal(0, 5, 96)
    corrected = apply_normalizing_flow(cal, preds)
    lower, upper, sigma = predictive_interval(cal, preds)
    for name, arr in [("corrected", corrected), ("lower", lower),
                      ("upper", upper), ("sigma", sigma)]:
        assert not np.any(np.isnan(arr)), f"NaN in {name}"


def test_model_saved_per_orbit_and_error_column():
    """
    The old code wrote every satellite to normalizing_flow_GEO_ClockError_ns.pt
    regardless of orbit type or error column, so 63 of 64 fits were lost.
    """
    train_normalizing_flow(RNG.normal(0, 1, 96),
                           orbit_type="MEO", error_col="EphemerisError_m")
    assert os.path.exists("models/saved/normalizing_flow_MEO_EphemerisError_m.pt"), \
        "flow not saved under its own orbit/error identity"
