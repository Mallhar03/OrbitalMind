"""
The scoring contract — the single admissible scorer for this project.

Read this before computing a W-statistic anywhere. Every W number that steers a
decision must come from this module and nowhere else. Five people computing five
W numbers with five implementations is exactly the failure this file exists to
stop.

Why this file exists
--------------------
The organisers' Note.pdf pins the metric with a reference dataset
(`data/SW_ReferenceData.xlsx`, 45 samples) and its expected result:

    Shapiro-Wilk W statistic : 0.9810
    p-value                  : 0.5840
    Hypothesis test result   : 0   (fail to reject H0; H0 = data is normal)
    Significance level alpha : 0.05

`scipy.stats.shapiro` (Royston's AS R94 Shapiro-Wilk) returns **0.985174** on
that same vector, so it does *not* reproduce the benchmark and must not be used
to score this task. The organisers' figures are quoted to three decimals (the
trailing zeros are formatting), and they are reproduced to that precision by the
**Shapiro-Francia** statistic using **Blom plotting positions** together with
the **Royston (1993)** p-value approximation:

    W = 0.981386  ->  0.981   (benchmark 0.9810)
    p = 0.583828  ->  0.584   (benchmark 0.5840)
    H = 0

`tests/test_scoring_contract.py` pins this against the shipped reference file.

The reported score
------------------
Per Note.pdf, the four residual parameters -- x_error, y_error, z_error and
satclockerror -- carry **equal weight**. The reported W, p and H are the mean
of the per-parameter values (H recomputed from the averaged p at alpha=0.05).
Priority 2 adds the residual mean and standard deviation; priority 3 the Q-Q
plot (see `evaluation.gaussian_check`). A confidence interval on the mean
residual is included per parameter.

Residual convention: residual = predicted - truth. W and p are invariant to
this sign; only the reported mean flips, and it is documented as predicted
minus truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

# The four residual parameters, in canonical order, each given equal weight.
PARAMETERS: tuple[str, ...] = ("x_error", "y_error", "z_error", "satclockerror")

# Significance level fixed by the organisers.
ALPHA: float = 0.05

# Shapiro-Francia's p-value approximation (Royston 1993) is defined on this
# range of sample sizes.
SF_MIN_N: int = 5
SF_MAX_N: int = 5000


@dataclass(frozen=True)
class ParameterScore:
    """The full scored result for one residual parameter.

    Attributes:
        parameter: parameter name, one of PARAMETERS
        n:         number of finite residuals scored
        W:         Shapiro-Francia W statistic (higher is better)
        p_value:   Royston (1993) p-value for the SF statistic
        H:         hypothesis result, 0 = fail to reject H0 (normal),
                   1 = reject H0, at ALPHA
        mean:      mean residual, metres (priority 2; predicted - truth)
        std:       residual standard deviation, metres (priority 2)
        ci_low:    lower bound of the 95% CI on the mean residual, metres
        ci_high:   upper bound of the 95% CI on the mean residual, metres
    """
    parameter: str
    n: int
    W: float
    p_value: float
    H: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ScoreResult:
    """The averaged, submission-ready score across all four parameters.

    Attributes:
        per_parameter: mapping parameter name -> ParameterScore
        W:             mean W across parameters (priority 1)
        p_value:       mean p across parameters (priority 1)
        H:             hypothesis result from the averaged p at ALPHA
    """
    per_parameter: dict[str, ParameterScore]
    W: float
    p_value: float
    H: int

    def as_dict(self) -> dict:
        """Return a plain-dict form suitable for JSON or a report writer."""
        return {
            "W": self.W,
            "p_value": self.p_value,
            "H": self.H,
            "per_parameter": {k: asdict(v) for k, v in self.per_parameter.items()},
        }


def shapiro_francia(x: Sequence[float]) -> tuple[float, float, int]:
    """Shapiro-Francia normality test — the project's admissible W statistic.

    W is the squared Pearson correlation between the ordered sample and the
    Blom approximation of standard-normal order-statistic medians. The p-value
    follows Royston (1993), "A Toolkit for Testing for Non-Normality in
    Complete and Censored Samples".

    This reproduces the organisers' reference benchmark exactly:
    W = 0.9810, p = 0.5840, H = 0 on data/SW_ReferenceData.xlsx.

    Args:
        x: 1-D sample of residuals. Non-finite values are dropped.
    Returns:
        (W, p_value, H) where H is 0 if p > ALPHA else 1.
    Raises:
        ValueError: if fewer than SF_MIN_N finite samples remain.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n < SF_MIN_N:
        raise ValueError(
            f"Shapiro-Francia needs at least {SF_MIN_N} finite samples, got {n}"
        )

    # A degenerate constant sample has no correlation to define; it is as
    # non-normal as a sample can be.
    if np.std(a) < 1e-15:
        return 0.0, 0.0, 1

    xs = np.sort(a)
    i = np.arange(1, n + 1)
    # Blom plotting positions -> approximate normal order-statistic medians.
    m = stats.norm.ppf((i - 0.375) / (n + 0.25))
    W = float(np.corrcoef(xs, m)[0, 1] ** 2)

    p = _sf_pvalue(W, n)
    H = 0 if p > ALPHA else 1
    return W, p, H


def _sf_pvalue(W: float, n: int) -> float:
    """Royston (1993) p-value for the Shapiro-Francia statistic.

    The normalising transform is only calibrated for SF_MIN_N <= n <= SF_MAX_N;
    outside that range the p-value is clamped to the nearest valid sample size
    so the statistic stays usable while staying honest that it is approximate.

    Args:
        W: Shapiro-Francia W statistic.
        n: sample size.
    Returns:
        Upper-tail p-value in [0, 1].
    """
    n_eff = int(min(max(n, SF_MIN_N), SF_MAX_N))
    u = np.log(n_eff)
    v = np.log(u)
    mu = -1.2725 + 1.0521 * (v - u)
    sigma = 1.0308 - 0.26758 * (v + 2.0 / u)
    # W == 1 would make log(1 - W) diverge; a perfect fit is unambiguously normal.
    if W >= 1.0:
        return 1.0
    z = (np.log(1.0 - W) - mu) / sigma
    return float(stats.norm.sf(z))


def _mean_ci(a: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided confidence interval on the mean of a sample (t-distribution).

    Args:
        a:          1-D finite sample.
        confidence: coverage, default 0.95.
    Returns:
        (low, high) bounds on the mean.
    """
    n = a.size
    mean = float(np.mean(a))
    if n < 2:
        return mean, mean
    sem = float(np.std(a, ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return mean - t * sem, mean + t * sem


def score_parameter(parameter: str, residuals: Sequence[float]) -> ParameterScore:
    """Score one parameter's residuals end to end.

    Args:
        parameter: parameter name (for labelling; need not be in PARAMETERS).
        residuals: 1-D residual sample (predicted - truth), metres.
    Returns:
        A ParameterScore.
    Raises:
        ValueError: if fewer than SF_MIN_N finite residuals remain.
    """
    a = np.asarray(residuals, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    W, p, H = shapiro_francia(a)
    ci_low, ci_high = _mean_ci(a)
    return ParameterScore(
        parameter=parameter,
        n=int(a.size),
        W=W,
        p_value=p,
        H=H,
        mean=float(np.mean(a)),
        std=float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def score_residuals(residuals_by_parameter: Mapping[str, Sequence[float]]) -> ScoreResult:
    """Score a full set of per-parameter residuals into the reported result.

    The averaged W and p give equal weight to each parameter present, per
    Note.pdf. H is recomputed from the averaged p at ALPHA.

    Args:
        residuals_by_parameter: mapping parameter name -> residual sample.
            Callers normally pass all of PARAMETERS.
    Returns:
        A ScoreResult.
    Raises:
        ValueError: if no parameters are supplied, or any parameter has fewer
            than SF_MIN_N finite residuals.
    """
    if not residuals_by_parameter:
        raise ValueError("no parameters supplied to score")

    per: dict[str, ParameterScore] = {}
    for name, resids in residuals_by_parameter.items():
        per[name] = score_parameter(name, resids)

    avg_W = float(np.mean([s.W for s in per.values()]))
    avg_p = float(np.mean([s.p_value for s in per.values()]))
    H = 0 if avg_p > ALPHA else 1
    return ScoreResult(per_parameter=per, W=avg_W, p_value=avg_p, H=H)
