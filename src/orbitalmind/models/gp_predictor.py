"""
Gaussian Process forecaster for GNSS satellite error prediction.

WHY GP
------
The harmonic model captures the dominant periodicities exactly, but the
residual after removing harmonics is not white noise — it has structured
autocorrelation at the 2-8 hour scale driven by atmospheric and orbit-
determination latency effects. A GP with a periodic + Matérn kernel
captures this additional structure while providing calibrated uncertainty
intervals.

KERNEL DESIGN
-------------
    k(t, t') =  C₁ × ExpSineSquared(l₁, T₁=43200)   # 12-h orbital
              + C₂ × ExpSineSquared(l₂, T₂=86400)   # 24-h diurnal
              + C₃ × Matern52(l₃)                   # mid-range correlation
              + WhiteKernel(σ²)                      # noise floor

This is the minimum kernel that captures all known GNSS error physics
without over-parameterising on 42-143 training points.

PERFORMANCE NOTE
----------------
sklearn GP with n_restarts_optimizer=2 trains in <5 s per column on CPU at
n≈100. Four columns × five series = 20 fits, total ≈ 100 s on any machine.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    Matern,
    WhiteKernel,
)

from orbitalmind.ingest import Series


def _default_kernel():
    """
    Construct the GNSS-motivated GP kernel.

    Returns a fresh kernel instance (sklearn kernels are stateful after fitting,
    so each GP needs its own copy).
    """
    periodic_12h = ConstantKernel(1.0, (1e-3, 1e2)) * ExpSineSquared(
        length_scale=1.0,
        periodicity=43_200.0,
        length_scale_bounds=(1e2, 1e5),
        periodicity_bounds="fixed",
    )
    periodic_24h = ConstantKernel(1.0, (1e-3, 1e2)) * ExpSineSquared(
        length_scale=1.0,
        periodicity=86_400.0,
        length_scale_bounds=(1e2, 1e5),
        periodicity_bounds="fixed",
    )
    medium_range = ConstantKernel(0.5, (1e-3, 1e2)) * Matern(
        length_scale=3_600.0,
        length_scale_bounds=(600.0, 7_200.0),
        nu=2.5,
    )
    noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
    return periodic_12h + periodic_24h + medium_range + noise


class GaussianProcessPredictor:
    """
    GP forecaster with periodic + Matérn kernel.

    Fits one GP per target column.  Implements the ``Predictor`` protocol
    from ``orbitalmind.interfaces`` (duck-typed).

    Attributes:
        n_restarts: number of optimizer restarts for hyperparameter marginal
            likelihood maximisation.  2 is sufficient for the PS-08 series
            sizes; increase to 5 if time permits.
    """

    def __init__(self, n_restarts: int = 2, random_state: int = 42) -> None:
        self.n_restarts = n_restarts
        self.random_state = random_state
        self._gps: list[GaussianProcessRegressor] | None = None
        self._t0: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, series: Series) -> "GaussianProcessPredictor":
        """
        Fit GP models to a training Series.

        Args:
            series: training Series from ``orbitalmind.ingest.load_series``.
        Returns:
            self — for method chaining.
        """
        self._t0 = series.times.iloc[0].to_pydatetime()
        t_sec = self._to_sec(series.times).reshape(-1, 1)
        Y = series.values()                              # (n, 4)

        self._gps = []
        for col_idx in range(Y.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=_default_kernel(),
                n_restarts_optimizer=self.n_restarts,
                normalize_y=True,
                random_state=self.random_state,
            )
            gp.fit(t_sec, Y[:, col_idx])
            self._gps.append(gp)

        return self

    def predict(self, t_query: Sequence[datetime]) -> np.ndarray:
        """
        Predict at arbitrary query timestamps.

        Args:
            t_query: sequence of ``datetime`` objects.
        Returns:
            ``np.ndarray`` of shape ``(N, 4)``, metres, in TARGET_COLUMNS order.
        Raises:
            RuntimeError: if ``fit()`` has not been called.
        """
        if self._gps is None or self._t0 is None:
            raise RuntimeError("Call fit() before predict()")

        t_sec = self._to_sec(t_query).reshape(-1, 1)
        cols = [gp.predict(t_sec) for gp in self._gps]
        return np.column_stack(cols)                    # (N, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_sec(self, times) -> np.ndarray:
        """Convert timestamps to float seconds elapsed since t0."""
        return np.array(
            [(t - self._t0).total_seconds() for t in times],
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        fitted = self._gps is not None
        return (
            f"GaussianProcessPredictor(n_restarts={self.n_restarts}, "
            f"fitted={fitted})"
        )
