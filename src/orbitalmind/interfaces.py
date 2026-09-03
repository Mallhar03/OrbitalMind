"""
The prediction contract — the interface every model lane codes against.

Read this before building or wiring any forecaster. The point of this module is
that the forecaster lane, the shaping lane and the delivery lane all bind to
this one signature instead of to each other's half-finished code. A lane can be
built and tested against `PersistencePredictor` before any real model exists.

The contract
------------
A predictor answers the organisers' evaluation question directly (Note.pdf):
given arbitrary 8th-day timestamps -- which may or may not be uniform -- return
the four error parameters at each timestamp, in metres.

    predict(t_query: sequence of datetimes) -> np.ndarray of shape (n, 4)

The four columns are, in order, TARGET_COLUMNS from `orbitalmind.ingest`:
x_error, y_error, z_error, satclockerror -- all in metres. This column order is
the same one `evaluation.scoring` and the (n, 4) truth array use, so a
prediction and its truth line up without any renaming.

Anything satisfying `Predictor` is scorable: build the residual as
prediction - truth per column and hand it to `evaluation.scoring.score_residuals`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from orbitalmind.ingest import TARGET_COLUMNS, PARAMETERS, Series

# Number of target columns a prediction must have.
N_TARGETS = len(TARGET_COLUMNS)


@runtime_checkable
class Predictor(Protocol):
    """Anything that can forecast the four errors at arbitrary timestamps.

    Implementations must return metres, in TARGET_COLUMNS order, one row per
    query timestamp.
    """

    def predict(self, t_query: Sequence[datetime]) -> np.ndarray:
        """Forecast the four errors at each query timestamp.

        Args:
            t_query: sequence of datetimes to predict at (need not be uniform,
                need not be sorted).
        Returns:
            Array of shape (len(t_query), 4), metres, in TARGET_COLUMNS order.
        """
        ...


def to_datetime_array(t_query: Sequence[datetime]) -> np.ndarray:
    """Coerce a timestamp sequence to a 1-D datetime64[ns] array.

    Accepts datetimes, strings, or a pandas DatetimeIndex/Series.

    Args:
        t_query: sequence of timestamps.
    Returns:
        1-D numpy datetime64[ns] array.
    """
    return pd.to_datetime(pd.Index(t_query)).to_numpy()


def predict_frame(predictor: Predictor, t_query: Sequence[datetime]) -> pd.DataFrame:
    """Run a predictor and return a labelled frame instead of a bare array.

    A convenience for the delivery lane; the core contract stays the (n, 4)
    array from `predict`.

    Args:
        predictor: any Predictor.
        t_query:   timestamps to predict at.
    Returns:
        DataFrame with a 'Timestamp' column and the four TARGET_COLUMNS.
    Raises:
        ValueError: if the predictor returns the wrong shape.
    """
    ts = to_datetime_array(t_query)
    preds = np.asarray(predictor.predict(t_query), dtype=np.float64)
    if preds.shape != (len(ts), N_TARGETS):
        raise ValueError(
            f"predictor returned {preds.shape}, expected {(len(ts), N_TARGETS)}"
        )
    frame = pd.DataFrame({"Timestamp": ts})
    for j, col in enumerate(TARGET_COLUMNS):
        frame[col] = preds[:, j]
    return frame


def residuals_by_parameter(
    predictor: Predictor, truth: Series
) -> dict[str, np.ndarray]:
    """Build per-parameter residuals (prediction - truth) for scoring.

    Predicts at the truth series' own timestamps and differences column by
    column, giving exactly the dict `evaluation.scoring.score_residuals` wants.

    Args:
        predictor: any Predictor.
        truth:     a held-out Series carrying the ground-truth values.
    Returns:
        Mapping short parameter name -> residual array (prediction - truth).
    """
    preds = np.asarray(predictor.predict(list(truth.times)), dtype=np.float64)
    if preds.shape != (truth.n, N_TARGETS):
        raise ValueError(
            f"predictor returned {preds.shape}, expected {(truth.n, N_TARGETS)}"
        )
    actual = truth.values()
    return {PARAMETERS[j]: preds[:, j] - actual[:, j] for j in range(N_TARGETS)}


@dataclass
class PersistencePredictor:
    """The reference stub: predict the last observed value at every timestamp.

    This is a real, deterministic Predictor -- not a mock -- so every downstream
    lane can be built and scored end to end before a trained model exists.
    Killing this stub is never necessary; a real model simply replaces it behind
    the same interface.

    Attributes:
        last_values: the four errors to repeat, metres, in TARGET_COLUMNS order.
    """
    last_values: np.ndarray

    @classmethod
    def from_series(cls, series: Series) -> "PersistencePredictor":
        """Build a persistence predictor from a training series' final row.

        Args:
            series: the (training) Series to anchor on.
        Returns:
            A PersistencePredictor repeating that series' last observed errors.
        """
        return cls(last_values=series.values()[-1].copy())

    def predict(self, t_query: Sequence[datetime]) -> np.ndarray:
        """Return the anchored last value at every query timestamp.

        Args:
            t_query: timestamps to predict at.
        Returns:
            Array of shape (len(t_query), 4).
        """
        n = len(to_datetime_array(t_query))
        return np.tile(self.last_values.reshape(1, N_TARGETS), (n, 1))
