"""
The shaping / calibration lane — bias, dispersion, and diagnostics.

Read this before touching it, because it is the part most likely to be
"improved" into fraud. It exists to deliver what the organisers ask for in
priorities 1-3 of Note.pdf — the residual mean, the confidence interval, the
dispersion, and the Q-Q diagnostic — NOT to raise the Shapiro-Francia W.

Why it CANNOT raise W, and does not even touch the forecast
-----------------------------------------------------------
1. At the real evaluation you submit PREDICTIONS. The organisers compute the
   residual themselves as (your prediction - their hidden truth) and run the
   test on that. You never hand them a residual, so any transform applied to a
   residual after the fact reaches nothing but your own internal scorer — pure
   self-deception. This module does not do it.
2. This layer does NOT modify the submitted point forecast at all. The point
   prediction is exactly the model's raw output; shaping only ATTACHES a
   predictive interval and the Q-Q diagnostic alongside it. So it is
   structurally impossible for shaping to change W or the predictions you
   submit. `tests/test_shaping.py` pins this.

Why there is no bias correction
-------------------------------
An earlier version added a per-parameter bias learned from a held-out training
tail. It was MEASURED on the shipped day-8 data and made the priority-2 residual
mean WORSE on 4 of 5 series (net mean |residual mean| 0.36 -> 0.62): a bias
fit on the calm training week does not transfer through day-8's regime change,
least of all the GEO divergence. Per the project's anti-clutter rule it did not
earn its place and was removed. If future data shows a bias that generalises off
the validation tail, reintroduce it behind a measured gate — not on faith.

What it DOES deliver
--------------------
* `sigma` — a per-parameter robust dispersion (1.4826 * MAD of OUT-OF-SAMPLE
  training-tail residuals) used to draw a predictive interval around each point.
* Q-Q diagnostics via `evaluation.gaussian_check`, the priority-3 deliverable.

All of it is fit on training data only. No test/query truth is read here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitalmind.ingest import Series, PARAMETERS
from orbitalmind.predict import _slice_series, _fit, VALIDATION_TAIL_FRACTION

N_PARAMS = len(PARAMETERS)


@dataclass(frozen=True)
class Calibration:
    """Per-parameter dispersion learned from out-of-sample training residuals.

    It carries no point correction: the submitted forecast is the model's raw
    output. This object only supplies the predictive interval around it.

    Attributes:
        sigma: shape (4,), robust dispersion (metres) for predictive intervals.
        source: 'validation' if fit on a held-out training tail, 'insample' if
               the series was too short to split (documented, not silent).
    """

    sigma: np.ndarray
    source: str

    def interval(
        self, predictions: np.ndarray, z: float = 1.96
    ) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric predictive interval around the (unmodified) point forecast.

        Args:
            predictions: (n, 4) point forecast — returned to the caller untouched.
            z: standard-normal multiplier (1.96 ≈ 95%).
        Returns:
            (low, high), each (n, 4), the point ± z * sigma.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        half = z * self.sigma.reshape(1, N_PARAMS)
        return preds - half, preds + half


def _robust_sigma(residuals: np.ndarray) -> float:
    """1.4826 * MAD — a heavy-tail-resistant standard-deviation estimate.

    Args:
        residuals: 1-D residual sample.
    Returns:
        Robust scale; falls back to np.std for a degenerate sample.
    """
    r = np.asarray(residuals, dtype=np.float64)
    if r.size < 2:
        return 0.0
    mad = np.median(np.abs(r - np.median(r)))
    sigma = 1.4826 * mad
    # A near-constant-but-not-identical sample can give MAD 0; fall back to std.
    return float(sigma) if sigma > 0 else float(np.std(r, ddof=1))


def calibrate(model_name: str, train: Series) -> Calibration:
    """Learn a leak-free predictive dispersion for one series.

    Fits the chosen model on the earlier part of the training record and reads
    its residual spread on the held-out training tail — exactly the split used
    for model selection, so nothing here sees the test/query truth.

    Args:
        model_name: the candidate name chosen by predict.select_model.
        train: the full training Series.
    Returns:
        A Calibration. If the series is too short to split, sigma is the
        in-sample robust spread, with source='insample'.
    """
    n = train.n
    cut = int(round(n * (1.0 - VALIDATION_TAIL_FRACTION)))
    if cut < 5 or (n - cut) < 4:
        # Too short to hold out a tail — refit on all, report in-sample spread.
        model = _fit(model_name, train)
        preds = np.asarray(model.predict(list(train.times)), dtype=np.float64)
        resid = train.values() - preds
        sigma = np.array([_robust_sigma(resid[:, j]) for j in range(N_PARAMS)])
        return Calibration(sigma=sigma, source="insample")

    fit_part = _slice_series(train, 0, cut)
    val_part = _slice_series(train, cut, n)
    model = _fit(model_name, fit_part)
    val_pred = np.asarray(model.predict(list(val_part.times)), dtype=np.float64)
    resid = val_part.values() - val_pred  # truth - prediction
    sigma = np.array([_robust_sigma(resid[:, j]) for j in range(N_PARAMS)])
    return Calibration(sigma=sigma, source="validation")
