"""
Tests for data-driven train/calibration/forecast window computation.

These windows used to be hardcoded (480/576/672), which silently produced
in-sample "day 8" predictions on a 672-row (7-day) file and discarded every
row past index 672 on longer files. compute_splits derives them from the
actual series length instead.
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.splits import compute_splits, MIN_LENGTH_MSG


def test_backtest_holdout_is_the_final_horizon():
    """The scored hold-out must be the last `horizon` steps of the record."""
    s = compute_splits(n=671)
    assert s.backtest.target == (575, 671)


def test_backtest_training_never_overlaps_holdout():
    """Training data must end before the calibration window begins."""
    s = compute_splits(n=671)
    assert s.backtest.train[1] <= s.backtest.cal[0]
    assert s.backtest.cal[1] <= s.backtest.target[0]


def test_submission_forecasts_beyond_the_record():
    """
    The submission window must start at the end of the data, not inside it.
    This is the day-7-vs-day-8 bug: with a 7-day (672-row) file there is no
    day 8 in the file, so the forecast has to extend past the final row.
    """
    s = compute_splits(n=671)
    assert s.submission.target == (671, 767)
    assert s.submission.target[0] == 671


def test_submission_input_is_the_most_recent_data():
    """Forecast input must be the last seq_len observed steps."""
    s = compute_splits(n=671, seq_len=96)
    assert s.submission.input == (575, 671)


def test_submission_calibration_is_out_of_sample():
    """Calibration residuals are worthless if the models trained on them."""
    s = compute_splits(n=671)
    assert s.submission.cal[0] >= s.submission.train[1]


def test_all_rows_are_used_on_a_long_record():
    """A 1343-step record must not silently discard its tail."""
    s = compute_splits(n=1343)
    assert s.submission.train[1] == 1343 - 96
    assert s.submission.target == (1343, 1439)


@pytest.mark.parametrize("n", [671, 767, 1343])
def test_windows_stay_in_bounds(n):
    """No window may index outside the available data."""
    s = compute_splits(n=n)
    for plan in (s.backtest, s.submission):
        for name in ("train", "cal", "input"):
            start, stop = getattr(plan, name)
            assert 0 <= start < stop <= n, f"{name} out of bounds: {(start, stop)}"


@pytest.mark.parametrize("n", [671, 767, 1343])
def test_every_model_gets_trainable_sequences(n):
    """
    TFT builds direct (input, target) pairs and needs at least
    seq_len + horizon training points to form even one.
    """
    s = compute_splits(n=n, seq_len=96, horizon=96)
    for plan in (s.backtest, s.submission):
        train_len = plan.train[1] - plan.train[0]
        assert train_len >= 96 + 96, f"only {train_len} training points"


def test_short_record_raises_a_clear_error():
    """Too-short input must fail loudly, not silently produce zeros."""
    with pytest.raises(ValueError) as exc:
        compute_splits(n=200)
    assert MIN_LENGTH_MSG in str(exc.value)


def test_horizon_length_is_exactly_the_forecast_length():
    """Every window that feeds a 96-step forecast must be 96 long."""
    s = compute_splits(n=671, horizon=96)
    assert s.backtest.target[1] - s.backtest.target[0] == 96
    assert s.submission.target[1] - s.submission.target[0] == 96
    assert s.backtest.cal[1] - s.backtest.cal[0] == 96


def test_calibration_input_precedes_the_calibration_window():
    """The forecast that produces calibration residuals must run from history."""
    s = compute_splits(n=671, seq_len=96)
    assert s.backtest.cal_input == (383, 479)
    assert s.submission.cal_input == (479, 575)


def test_meta_and_flow_calibrate_on_disjoint_halves():
    """
    The flow must fit residuals the meta-learner never saw, otherwise the
    learned predictive spread is optimistic.
    """
    s = compute_splits(n=671)
    for plan in (s.backtest, s.submission):
        assert plan.cal_meta[1] == plan.cal_flow[0]
        assert plan.cal_meta[0] == plan.cal[0]
        assert plan.cal_flow[1] == plan.cal[1]
        assert plan.cal_meta[1] - plan.cal_meta[0] > 0
        assert plan.cal_flow[1] - plan.cal_flow[0] > 0


@pytest.mark.parametrize("n", [671, 767, 1343])
def test_calibration_never_leaks_into_training(n):
    """Base models must not have trained on anything they are calibrated against."""
    s = compute_splits(n=n)
    for plan in (s.backtest, s.submission):
        assert plan.train[1] <= plan.cal[0], "calibration window overlaps training"
