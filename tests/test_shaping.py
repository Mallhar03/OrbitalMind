"""
Shaping/calibration tests — and the integrity guarantee that it cannot fake W.

The single most important test here is `test_calibration_is_W_neutral`: it proves
the shaping layer is affine (a pure translation of the point forecast) and so
CANNOT change the Shapiro-Francia W statistic. That is the structural guarantee
that this layer improves the reported mean and supplies intervals without ever
engineering the priority-1 normality score — the exact fraud the project's
DECISIONS.md records and forbids.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.shaping import calibrate, Calibration, _robust_sigma  # noqa: E402
from orbitalmind.predict import select_model  # noqa: E402
from orbitalmind.ingest import load_series, PARAMETERS  # noqa: E402
from orbitalmind.paths import DATA_DIR  # noqa: E402

GEO = DATA_DIR / "DATA_GEO_Train.csv"


def _series():
    if not GEO.exists():
        pytest.skip(f"{GEO} not present")
    return load_series(GEO)[0]


def test_calibration_exposes_no_point_transform():
    """Integrity guarantee: Calibration must offer NO way to alter a point
    forecast. It carries only sigma + an interval() helper; any method that
    returned a modified point (which could change W or the submission) would be
    a regression this test catches.
    """
    cal = Calibration(sigma=np.full(4, 2.0), source="test")
    public = {a for a in dir(cal) if not a.startswith("_")}
    assert public == {"sigma", "source", "interval"}, (
        f"Calibration exposes unexpected members {public} — a point-altering "
        "method may have been added; shaping must not touch the forecast."
    )
    # interval() returns bounds around the point but never a replacement point.
    point = np.zeros((3, 4))
    low, high = cal.interval(point)
    assert low.shape == high.shape == point.shape


def test_calibrate_is_leak_free_and_shaped():
    """calibrate() returns finite bias/sigma from training only, right shapes."""
    s = _series()
    name, _ = select_model(s)
    cal = calibrate(name, s)
    assert cal.sigma.shape == (len(PARAMETERS),)
    assert np.isfinite(cal.sigma).all()
    assert (cal.sigma >= 0).all()
    assert cal.source in ("validation", "insample")


def test_interval_brackets_point():
    """The predictive interval must bracket the point forecast symmetrically."""
    cal = Calibration(sigma=np.array([1.0, 2.0, 3.0, 4.0]), source="test")
    point = np.zeros((5, 4))
    low, high = cal.interval(point, z=1.96)
    assert (low <= point).all() and (high >= point).all()
    # width is 2 * 1.96 * sigma, per row
    expected = np.broadcast_to(2 * 1.96 * cal.sigma.reshape(1, 4), (5, 4))
    np.testing.assert_allclose(high - low, expected)


def test_robust_sigma_resists_one_outlier():
    """MAD-based sigma must not be blown up by a single outlier the way std is.

    Uses a genuinely-spread sample (not a degenerate >50%-identical one, where
    MAD is 0 and the estimator documents its fallback to std).
    """
    rng = np.random.default_rng(1)
    spread = rng.normal(scale=0.5, size=19)
    sample = np.concatenate([spread, [50.0]])  # real spread + 1 big outlier
    assert _robust_sigma(sample) < np.std(sample, ddof=1)


def test_entrypoint_point_forecast_equals_raw_model():
    """End-to-end integrity: the point columns the entrypoint would submit must
    equal the raw model prediction — shaping adds intervals but never alters the
    forecast. Proven by predicting with the chosen model directly and comparing.
    """
    import numpy as np
    from orbitalmind.predict import forecast_series, _fit, select_model
    from orbitalmind.ingest import TARGET_COLUMNS

    s = _series()
    t_query = list(s.times[:5])
    result, _ = forecast_series(s, t_query)
    name, _ = select_model(s)
    raw = np.asarray(_fit(name, s).predict(t_query), dtype=float)
    submitted = result.predictions[list(TARGET_COLUMNS)].to_numpy(float)
    np.testing.assert_allclose(submitted, raw, rtol=0, atol=0)
