"""
Scoring contract tests — pin the scorer to the organisers' benchmark.

The one test that matters most is `test_reference_benchmark_reproduced`: it runs
the shipped reference vector (data/SW_ReferenceData.xlsx) through the scorer and
asserts the organisers' published result to the stated precision:

    W = 0.9810, p = 0.5840, H = 0.

If that assertion ever fails, no W number this project reports can be trusted,
because it is no longer computing the same statistic the evaluator computes.
scipy.stats.shapiro returns 0.9851 on the same vector, which is why it is not
used and why this test exists.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.evaluation.scoring import (  # noqa: E402
    shapiro_francia, score_parameter, score_residuals, PARAMETERS, ALPHA,
)
from orbitalmind.paths import DATA_DIR  # noqa: E402

REFERENCE_XLSX = DATA_DIR / "SW_ReferenceData.xlsx"


def _reference_vector() -> np.ndarray:
    """The 45-sample benchmark vector, header row included as a value."""
    if not REFERENCE_XLSX.exists():
        pytest.skip(f"{REFERENCE_XLSX} not present")
    return pd.read_excel(REFERENCE_XLSX, header=None)[0].to_numpy(dtype=float)


def test_reference_benchmark_reproduced():
    """The scorer reproduces the organisers' published benchmark result.

    Note.pdf states W=0.9810, p=0.5840, H=0 for the reference dataset. Those
    figures are quoted to three decimal places (the trailing zero is
    formatting), and the Shapiro-Francia statistic with Blom plotting positions
    reproduces both to that precision: W=0.981386 -> 0.981, p=0.583828 -> 0.584.
    """
    x = _reference_vector()
    assert x.size == 45, "reference vector should hold 45 samples"

    W, p, H = shapiro_francia(x)
    assert round(W, 3) == 0.981, f"W={W} does not round to the benchmark 0.981"
    assert round(p, 3) == 0.584, f"p={p} does not round to the benchmark 0.584"
    assert H == 0, "benchmark must fail to reject normality"


def test_scipy_shapiro_would_not_match():
    """Documents why scipy.stats.shapiro is not the admissible scorer.

    Royston's Shapiro-Wilk gives ~0.9852 on the reference vector, which is not
    the organisers' 0.9810. If a future scipy ever converged on the SF value
    this test would flag that the distinction had changed.
    """
    from scipy import stats
    x = _reference_vector()
    W_royston = float(stats.shapiro(x)[0])
    assert abs(W_royston - 0.9810) > 1e-3, (
        "scipy.shapiro unexpectedly matches the SF benchmark"
    )


def test_perfect_normal_scores_near_one_and_passes():
    """A clean Gaussian sample scores high W and fails to reject normality."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    W, p, H = shapiro_francia(x)
    assert W > 0.99
    assert H == 0


def test_strongly_nonnormal_is_rejected():
    """A heavy-tailed / skewed sample is rejected (H=1) with low W."""
    rng = np.random.default_rng(42)
    x = rng.exponential(scale=1.0, size=500)
    W, p, H = shapiro_francia(x)
    assert H == 1, "an exponential sample should reject normality"
    assert p < ALPHA


def test_constant_sample_is_degenerate_not_normal():
    """A zero-variance sample is treated as non-normal, never a spurious pass."""
    W, p, H = shapiro_francia(np.full(20, 3.14))
    assert W == 0.0 and p == 0.0 and H == 1


def test_too_few_samples_raises():
    """Shapiro-Francia is undefined below five samples and must not guess."""
    with pytest.raises(ValueError):
        shapiro_francia([1.0, 2.0, 3.0, 4.0])


def test_w_and_p_are_sign_invariant():
    """Flipping the residual sign leaves W and p unchanged (only the mean flips)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    W1, p1, _ = shapiro_francia(x)
    W2, p2, _ = shapiro_francia(-x)
    assert W1 == pytest.approx(W2)
    assert p1 == pytest.approx(p2)


def test_score_residuals_averages_equal_weight():
    """The reported W is the plain mean of the per-parameter W values."""
    rng = np.random.default_rng(7)
    resids = {p: rng.standard_normal(120) for p in PARAMETERS}
    result = score_residuals(resids)

    assert set(result.per_parameter) == set(PARAMETERS)
    expected_W = float(np.mean([s.W for s in result.per_parameter.values()]))
    assert result.W == pytest.approx(expected_W)
    assert result.H in (0, 1)


def test_parameter_score_reports_priority2_and_ci():
    """A ParameterScore carries mean, std and a bracketing CI (priorities 2/3)."""
    rng = np.random.default_rng(1)
    a = rng.normal(loc=2.0, scale=0.5, size=300)
    s = score_parameter("x_error", a)
    assert s.n == 300
    assert s.mean == pytest.approx(2.0, abs=0.1)
    assert s.std == pytest.approx(0.5, abs=0.1)
    assert s.ci_low < s.mean < s.ci_high


def test_small_block_scores_without_error():
    """The council's smallest real blocks (n=5, 6) must score, not crash."""
    rng = np.random.default_rng(3)
    for n in (5, 6):
        W, p, H = shapiro_francia(rng.standard_normal(n))
        assert 0.0 <= W <= 1.0
        assert 0.0 <= p <= 1.0
