"""
Harmonic regression forecaster for GNSS satellite error prediction.

PHYSICAL BASIS
--------------
GNSS satellite orbit and clock errors have strong periodicities at 12-hour
(half-orbital period) and 24-hour (full sidereal/solar day) cycles, driven
by the satellite's orbital mechanics and the diurnal variation of ionospheric
and tropospheric delays. This model fits those periodicities explicitly using
ordinary least squares, which is exact, deterministic, and never overfits on
the small PS-08 training series (42-143 rows per series).

MODEL FORM
----------
    y(t) = a_0                          (bias)
           + a_1 * t                    (linear drift)
           + sum_{k=1}^{K} [            (harmonics)
               A_k * cos(2π k t / T_1)
             + B_k * sin(2π k t / T_1)
           ]
           + sum_{k=1}^{K} [
               C_k * cos(2π k t / T_2)
             + D_k * sin(2π k t / T_2)
           ]

with T_1 = 43 200 s (12 h) and T_2 = 86 400 s (24 h), fitted independently
per target column (x_error, y_error, z_error, satclockerror) via lstsq.

WHY NOT A DEEP MODEL
--------------------
Each training series has 42-143 rows. A model with enough parameters to learn
arbitrary patterns would have more free parameters than data points. The
harmonic form reduces the free parameters to 2 + 4*K per target (2K per
period × K harmonics), which is ~14 parameters at K=3. That is always
well-determined on 42+ rows, and the physics guarantee the chosen basis is
exactly the right one.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from orbitalmind.ingest import Series, TARGET_COLUMNS

# Dominant GNSS orbital periods (seconds)
ORBITAL_PERIODS: tuple[float, ...] = (43_200.0, 86_400.0)  # 12 h, 24 h

# Number of harmonic components per period.  3 components × 2 periods = 6 pairs,
# plus bias + linear trend = 14 parameters per column.  Well-conditioned at n≥42.
N_HARMONICS: int = 3


class HarmonicPredictor:
    """
    Physics-informed harmonic regression forecaster.

    Fits an independent OLS harmonic model for each of the four target
    columns.  Implements the ``Predictor`` protocol from
    ``orbitalmind.interfaces`` (duck-typed — no import cycle needed).

    Attributes:
        n_harmonics: number of harmonic components per orbital period
        periods_sec: orbital periods to model, in seconds
    """

    def __init__(
        self,
        n_harmonics: int = N_HARMONICS,
        periods_sec: tuple[float, ...] = ORBITAL_PERIODS,
    ) -> None:
        self.n_harmonics = n_harmonics
        self.periods_sec = periods_sec
        self._coeffs: list[np.ndarray] | None = None  # (n_params,) per target col
        self._t0: datetime | None = None              # reference epoch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, series: Series) -> "HarmonicPredictor":
        """
        Fit harmonic model to a training Series.

        Args:
            series: training Series from ``orbitalmind.ingest.load_series``.
        Returns:
            self — for method chaining.
        """
        self._t0 = series.times.iloc[0].to_pydatetime()
        t_sec = self._timestamps_to_seconds(series.times)
        X = self._design_matrix(t_sec)                # (n, p)
        Y = series.values()                             # (n, 4)

        self._coeffs = []
        for col_idx in range(Y.shape[1]):
            coeffs, _, _, _ = np.linalg.lstsq(X, Y[:, col_idx], rcond=None)
            self._coeffs.append(coeffs)

        return self

    def predict(self, t_query: Sequence[datetime]) -> np.ndarray:
        """
        Predict at arbitrary query timestamps.

        Args:
            t_query: sequence of ``datetime`` objects (need not be uniform).
        Returns:
            ``np.ndarray`` of shape ``(N, 4)``, metres, in ``TARGET_COLUMNS``
            order: x_error, y_error, z_error, satclockerror.
        Raises:
            RuntimeError: if ``fit()`` has not been called.
        """
        if self._coeffs is None or self._t0 is None:
            raise RuntimeError("Call fit() before predict()")

        t_sec = np.array(
            [(t - self._t0).total_seconds() for t in t_query],
            dtype=np.float64,
        )
        X = self._design_matrix(t_sec)                # (N, p)
        cols = [X @ c for c in self._coeffs]           # list of (N,)
        return np.column_stack(cols)                   # (N, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _timestamps_to_seconds(self, times: "pd.Series") -> np.ndarray:
        """Convert a pandas Series of timestamps to float seconds from t0."""
        return np.array(
            [(t - self._t0).total_seconds() for t in times],
            dtype=np.float64,
        )

    def _design_matrix(self, t_sec: np.ndarray) -> np.ndarray:
        """
        Build the harmonic design matrix.

        Columns: [1, t, cos(ω₁t), sin(ω₁t), …, cos(Kω₁t), sin(Kω₁t),
                         cos(ω₂t), sin(ω₂t), …, cos(Kω₂t), sin(Kω₂t)]
        """
        cols: list[np.ndarray] = [np.ones_like(t_sec), t_sec]
        for T in self.periods_sec:
            for k in range(1, self.n_harmonics + 1):
                angle = 2.0 * np.pi * k * t_sec / T
                cols.append(np.cos(angle))
                cols.append(np.sin(angle))
        return np.column_stack(cols)                   # (n, 2 + 4*K)

    def __repr__(self) -> str:
        fitted = self._t0 is not None
        return (
            f"HarmonicPredictor(n_harmonics={self.n_harmonics}, "
            f"periods={self.periods_sec}, fitted={fitted})"
        )
