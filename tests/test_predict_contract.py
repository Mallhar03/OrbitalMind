"""
Prediction contract tests — pin the predict() interface and the reference stub.

These prove that a lane can be built and scored end to end against
`PersistencePredictor` with no trained model present: the stub satisfies the
`Predictor` protocol, returns (n, 4) at arbitrary timestamps, and flows straight
into the scorer. Every model lane binds to the same signature this locks down.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.interfaces import (  # noqa: E402
    Predictor, PersistencePredictor, predict_frame, residuals_by_parameter,
    N_TARGETS,
)
from orbitalmind.ingest import load_series, Series, TARGET_COLUMNS  # noqa: E402
from orbitalmind.evaluation.scoring import score_residuals  # noqa: E402
from orbitalmind.paths import DATA_DIR  # noqa: E402

GEO = DATA_DIR / "DATA_GEO_Train.csv"


def _series() -> Series:
    if not GEO.exists():
        pytest.skip(f"{GEO} not present")
    return load_series(GEO)[0]


def _arbitrary_timestamps(n=7):
    """Deliberately non-uniform, unsorted 8th-day timestamps."""
    base = datetime(2025, 9, 8, 0, 0, 0)
    offsets = [0, 137, 300, 301, 900, 45, 1439]  # minutes; jumbled, non-uniform
    return [base + pd.Timedelta(minutes=offsets[i % len(offsets)]) for i in range(n)]


def test_stub_satisfies_the_protocol():
    """PersistencePredictor is a structural Predictor."""
    s = _series()
    pred = PersistencePredictor.from_series(s)
    assert isinstance(pred, Predictor)


def test_predict_returns_n_by_four_at_arbitrary_timestamps():
    """The core contract: (len(t_query), 4) at non-uniform, unsorted times."""
    pred = PersistencePredictor.from_series(_series())
    ts = _arbitrary_timestamps(11)
    out = np.asarray(pred.predict(ts))
    assert out.shape == (11, N_TARGETS)
    assert np.isfinite(out).all()


def test_persistence_repeats_last_observed_value():
    """Every predicted row equals the anchor series' final observed errors."""
    s = _series()
    pred = PersistencePredictor.from_series(s)
    out = pred.predict(_arbitrary_timestamps(4))
    expected = s.values()[-1]
    assert np.allclose(out, np.tile(expected, (4, 1)))


def test_predict_frame_is_labelled_and_ordered():
    """predict_frame yields Timestamp + the four target columns, in order."""
    pred = PersistencePredictor.from_series(_series())
    ts = _arbitrary_timestamps(5)
    frame = predict_frame(pred, ts)
    assert list(frame.columns) == ["Timestamp", *TARGET_COLUMNS]
    assert len(frame) == 5


def test_stub_flows_end_to_end_into_the_scorer():
    """A stub prediction scores against a truth series with no model involved.

    This is the whole point of Phase 0: the delivery and shaping lanes can run
    the full predict -> residual -> score path today.
    """
    truth = _series()
    pred = PersistencePredictor.from_series(truth)
    resids = residuals_by_parameter(pred, truth)
    assert set(resids) == set(k.replace(" (m)", "") for k in TARGET_COLUMNS)
    result = score_residuals(resids)
    assert 0.0 <= result.W <= 1.0
    assert result.H in (0, 1)


def test_wrong_shape_predictor_is_rejected():
    """A predictor returning the wrong shape fails loudly, not silently."""
    class BadPredictor:
        def predict(self, t_query):
            return np.zeros((len(t_query), 3))  # only 3 columns

    with pytest.raises(ValueError):
        predict_frame(BadPredictor(), _arbitrary_timestamps(3))
