"""
Tests for compute_rmse_horizons: cumulative vs terminal RMSE.

The cumulative keys ('24hr' and friends) average the error over every step up
to the horizon, so a whole-day figure is pulled down by the 95 easier steps
before it. That hides long-horizon failure, which is the thing this project
most needs to see. The terminal keys ('at_24hr' and friends) report the error
at the horizon step alone.

The expected numbers below are hard-coded from a closed form derived by hand,
not read back out of the implementation, so they are a genuine regression
guard: with the error at step k made exactly k, the cumulative RMSE over steps
1..k is sqrt(mean(1^2 + ... + k^2)) = sqrt((k+1)(2k+1)/6).
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbitalmind.models.base_trainer import compute_rmse_horizons


CUMULATIVE_KEYS = ["15min", "30min", "1hr", "2hr", "12hr", "24hr"]
TERMINAL_KEYS = ["at_15min", "at_30min", "at_1hr", "at_2hr", "at_12hr", "at_24hr"]
HORIZON_STEPS = [1, 2, 4, 8, 48, 96]


@pytest.fixture
def ramp():
    """
    96 steps whose error at step k is exactly k.

    Error growing with lead time is the realistic case and the one that makes
    the cumulative/terminal distinction matter.
    """
    y_true = np.zeros(96)
    y_pred = np.arange(1, 97, dtype=float)
    return y_true, y_pred


def test_both_families_present_in_exact_order(ramp):
    """All six cumulative keys come first, then all six terminal keys."""
    result = compute_rmse_horizons(*ramp)
    assert list(result.keys()) == CUMULATIVE_KEYS + TERMINAL_KEYS


def test_cumulative_values_are_unchanged(ramp):
    """
    The five original horizons must return exactly what they always have.
    run_pipeline.py, ablation.py and several model tests compare against
    these numbers, so their meaning cannot shift.
    """
    result = compute_rmse_horizons(*ramp)
    assert result["15min"] == pytest.approx(1.0)
    assert result["30min"] == pytest.approx(1.5811388300841898)
    assert result["1hr"] == pytest.approx(2.7386127875258306)
    assert result["2hr"] == pytest.approx(5.049752469181039)
    assert result["24hr"] == pytest.approx(55.85845206113992)


def test_new_cumulative_12hr_follows_the_same_rule(ramp):
    """'12hr' is a prefix RMSE over steps 1..48, like its siblings."""
    result = compute_rmse_horizons(*ramp)
    assert result["12hr"] == pytest.approx(28.145455524234578)


def test_terminal_is_the_error_at_that_step_alone(ramp):
    """With the error at step k set to k, each terminal key returns k."""
    result = compute_rmse_horizons(*ramp)
    for key, step in zip(TERMINAL_KEYS, HORIZON_STEPS):
        assert result[key] == pytest.approx(float(step)), key


def test_terminal_exceeds_cumulative_when_error_grows(ramp):
    """
    The property that makes the terminal metric worth having: when error grows
    with lead time, the cumulative figure understates it. At 24hr the gap is
    55.9 against 96.0 -- the cumulative number is barely half the real
    end-of-day error.
    """
    result = compute_rmse_horizons(*ramp)
    for cum, term in zip(CUMULATIVE_KEYS[1:], TERMINAL_KEYS[1:]):
        assert result[term] > result[cum], f"{term} should exceed {cum}"
    assert result["at_24hr"] > 1.7 * result["24hr"]


def test_single_step_horizon_agrees_in_both_families(ramp):
    """At one step there is nothing to average, so the two must coincide."""
    result = compute_rmse_horizons(*ramp)
    assert result["at_15min"] == pytest.approx(result["15min"])


def test_perfect_prediction_is_zero_everywhere():
    """No error at any step means 0.0 for all twelve keys, not NaN."""
    y = np.linspace(-3.0, 7.5, 96)
    result = compute_rmse_horizons(y, y.copy())
    assert len(result) == 12
    for key, val in result.items():
        assert val == pytest.approx(0.0), key


def test_12hr_sits_at_step_48_exactly():
    """
    A single spike at index 47 is seen by the 12hr keys and by nothing
    shorter; move it one step earlier and 'at_12hr' must go silent. This pins
    the off-by-one in both directions.
    """
    y_true = np.zeros(96)

    on_48 = np.zeros(96)
    on_48[47] = 5.0
    result = compute_rmse_horizons(y_true, on_48)
    assert result["at_12hr"] == pytest.approx(5.0)
    assert result["at_2hr"] == pytest.approx(0.0)
    assert result["at_24hr"] == pytest.approx(0.0)
    # The cumulative view spreads that same spike over 48 steps.
    assert result["12hr"] == pytest.approx(0.7216878364870323)
    assert result["2hr"] == pytest.approx(0.0)

    on_47 = np.zeros(96)
    on_47[46] = 5.0
    shifted = compute_rmse_horizons(y_true, on_47)
    assert shifted["at_12hr"] == pytest.approx(0.0)
    assert shifted["12hr"] == pytest.approx(0.7216878364870323)


def test_24hr_sits_at_step_96_exactly():
    """The last predicted point, and only it, drives 'at_24hr'."""
    y_true = np.zeros(96)
    y_pred = np.zeros(96)
    y_pred[95] = 5.0
    result = compute_rmse_horizons(y_true, y_pred)
    assert result["at_24hr"] == pytest.approx(5.0)
    assert result["at_12hr"] == pytest.approx(0.0)
    assert result["24hr"] == pytest.approx(0.5103103630798288)


def test_short_array_cumulative_uses_what_is_available():
    """
    Existing behaviour, preserved: a cumulative horizon longer than the data
    averages over the whole array. With 4 steps, '24hr' equals '1hr'.
    """
    y_true = np.zeros(4)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_rmse_horizons(y_true, y_pred)
    assert result["1hr"] == pytest.approx(2.7386127875258306)
    assert result["2hr"] == pytest.approx(2.7386127875258306)
    assert result["12hr"] == pytest.approx(2.7386127875258306)
    assert result["24hr"] == pytest.approx(2.7386127875258306)


def test_short_array_terminal_is_nan():
    """
    Documented behaviour: a terminal horizon beyond the data is NaN, not the
    last available step. Clamping would report the step-4 error as 'at_24hr',
    a small plausible number a reader would take for a real 24-hour figure.
    NaN cannot be misread that way.
    """
    y_true = np.zeros(4)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_rmse_horizons(y_true, y_pred)
    assert result["at_15min"] == pytest.approx(1.0)
    assert result["at_30min"] == pytest.approx(2.0)
    assert result["at_1hr"] == pytest.approx(4.0)
    for key in ["at_2hr", "at_12hr", "at_24hr"]:
        assert math.isnan(result[key]), f"{key} should be NaN, got {result[key]}"


def test_short_array_terminal_never_clamps_to_a_flattering_value():
    """
    The failure mode NaN exists to prevent: a 4-step array whose error is tiny
    must not report that tiny number as the 24-hour error.
    """
    y_true = np.zeros(4)
    y_pred = np.full(4, 0.001)
    result = compute_rmse_horizons(y_true, y_pred)
    assert math.isnan(result["at_24hr"])


def test_length_taken_from_the_shorter_of_the_two_arrays():
    """A longer y_pred does not unlock horizons the truth cannot score."""
    result = compute_rmse_horizons(np.zeros(4), np.arange(1, 97, dtype=float))
    assert result["at_1hr"] == pytest.approx(4.0)
    assert math.isnan(result["at_24hr"])


def test_two_dimensional_input_is_flattened():
    """
    Existing flatten() behaviour is kept, and the terminal keys stay correct
    under it: a terminal RMSE over one element is the absolute error.
    """
    y_true = np.zeros((2, 3))
    y_pred = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])
    result = compute_rmse_horizons(y_true, y_pred)
    assert result["at_15min"] == pytest.approx(1.0)
    assert result["at_30min"] == pytest.approx(2.0)   # absolute value of -2.0
    assert result["at_1hr"] == pytest.approx(4.0)
    assert math.isnan(result["at_2hr"])
    assert result["30min"] == pytest.approx(math.sqrt(2.5))


def test_list_input_is_accepted():
    """Callers pass plain lists as well as arrays."""
    result = compute_rmse_horizons([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0])
    assert result["at_1hr"] == pytest.approx(4.0)
    assert result["1hr"] == pytest.approx(2.7386127875258306)


def test_every_value_is_a_plain_float(ramp):
    """The report writer formats these with %f; numpy scalars are not returned."""
    result = compute_rmse_horizons(*ramp)
    for key, val in result.items():
        assert type(val) is float, key


def test_short_array_produces_no_numpy_warnings():
    """
    The NaN for an out-of-reach terminal horizon is returned by an explicit
    guard, not by letting numpy average an empty slice. Both routes give NaN,
    but the unguarded one emits 'Mean of empty slice' twice per horizon, which
    would flood any run that scores a short series. This pins the guard.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compute_rmse_horizons(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert math.isnan(result["at_24hr"])
