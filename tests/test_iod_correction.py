"""
Tests for IOD (Issue of Data) jump correction.

Two defects found in iteration 9, both invisible until the pipeline was run on
real IGS data in the organisers' 672-row format.

  * The threshold was a fixed 2.0 ns. A GPS satellite whose clock drifts
    faster than 2 ns per 15-minute step therefore had EVERY step classified
    as an IOD jump: G02 and G03 both reported 671 jumps out of 671 possible
    steps. The correction then subtracted the whole trend, leaving nothing to
    model. An IOD upload is a discontinuity relative to a satellite's OWN
    step-to-step behaviour, so the threshold has to be derived from that
    behaviour rather than fixed in advance.

  * Correction shifts the series into its own coordinate frame. Reconstructed
    forecasts must be returned to the observed frame or they are wrong by the
    total accumulated offset -- 3339 ns for G02, 18216 ns for G03.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.preprocessing.iod_correction import correct_iod_jumps, count_jumps


def _ramp(slope: float, n: int = 672) -> pd.Series:
    """A clean linear clock drift with no discontinuity."""
    return pd.Series(np.arange(n, dtype=float) * slope)


def test_fast_drift_is_not_mistaken_for_jumps():
    """
    The G02/G03 defect. A steady 5 ns/step drift is normal behaviour for some
    GPS clocks and must not register as 671 separate IOD uploads.
    """
    series = _ramp(slope=5.0)
    assert count_jumps(series) == 0, "steady drift misclassified as IOD jumps"


def test_very_fast_drift_is_not_mistaken_for_jumps():
    """G03 drifts about 27 ns per step."""
    series = _ramp(slope=27.0)
    assert count_jumps(series) == 0


def test_steady_drift_survives_correction():
    """Correcting a jump-free series must leave its trend intact."""
    series = _ramp(slope=5.0)
    corrected = correct_iod_jumps(series)
    assert corrected.iloc[-1] == pytest.approx(series.iloc[-1], rel=1e-6), \
        "correction destroyed the trend of a jump-free series"


def test_a_real_discontinuity_is_still_removed():
    """A genuine step change on top of normal drift must be corrected out."""
    series = _ramp(slope=1.0)
    series.iloc[300:] += 500.0                      # IOD upload
    corrected = correct_iod_jumps(series)
    steps = corrected.diff().dropna().abs()
    assert steps.max() < 10.0, \
        f"discontinuity survived correction (max step {steps.max():.2f})"


def test_jump_detected_where_it_actually_is():
    series = _ramp(slope=1.0)
    series.iloc[300:] += 500.0
    assert count_jumps(series) == 1


def test_noise_alone_does_not_trigger_corrections():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0.0, 1.0, 672))
    assert count_jumps(series) <= 3, "Gaussian noise triggering IOD corrections"


def test_constant_series_is_safe():
    """Zero spread must not divide by zero or flag every step."""
    series = pd.Series(np.full(672, 7.0))
    assert count_jumps(series) == 0
    assert correct_iod_jumps(series).iloc[-1] == pytest.approx(7.0)


def test_index_is_preserved():
    series = _ramp(slope=3.0)
    series.index = pd.RangeIndex(100, 100 + len(series))
    corrected = correct_iod_jumps(series)
    assert list(corrected.index) == list(series.index)


def test_explicit_threshold_still_honoured():
    """An explicit threshold overrides the adaptive one, for callers that
    genuinely know the scale of their signal."""
    series = _ramp(slope=1.0)
    series.iloc[300:] += 500.0
    corrected = correct_iod_jumps(series, threshold_ns=100.0)
    assert corrected.diff().dropna().abs().max() < 10.0
