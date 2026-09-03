"""
Tests for scripts/score_day8.py — the committed day-8 scorer.

Everything here runs on synthetic frames built in the test itself. The real
held-out file data/raw/gnss_holdout.csv is never opened, read, sampled or
referenced by these tests, and it must not be: the scorer is developed against
fixtures precisely so that writing it cannot become a way of looking at the
answer.

The fixtures are deliberately exact rather than random. A perfect forecast must
score 0.0, a forecast displaced by a known constant must score that constant,
and a linear history must let the linear baseline score 0.0 — each of which
fails loudly if the horizon slicing, the join, or the baseline windows are
wrong by even one step.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import score_day8
from score_day8 import (
    IncompleteJoinError, align_predictions, baseline_forecasts, horizon_steps,
    load_input, load_submission, load_truth, order_keys, score, render_report,
)
from orbitalmind.models.base_trainer import compute_rmse_horizons

STEP     = pd.Timedelta(minutes=15)
HIST_END = pd.Timestamp("2026-01-07 23:45:00")
SATS     = ["S01", "S02"]


def _history_times(n_hist: int) -> pd.DatetimeIndex:
    """Timestamps for n_hist input epochs ending at HIST_END."""
    return pd.date_range(end=HIST_END, periods=n_hist, freq=STEP)


def _forecast_times(n_steps: int) -> pd.DatetimeIndex:
    """Timestamps for the n_steps forecast epochs following HIST_END."""
    return pd.date_range(start=HIST_END + STEP, periods=n_steps, freq=STEP)


def make_input(sats=SATS, n_hist=96, clock_slope=2.0, clock_start=100.0,
               eph_slope=0.0, eph_start=5.0) -> pd.DataFrame:
    """
    Build a 7-day-style input frame with an exactly linear history per satellite.

    Args:
        sats:        satellite ids
        n_hist:      epochs per satellite
        clock_slope: ns added per 15-minute step
        clock_start: first clock value
        eph_slope:   metres added per step
        eph_start:   first ephemeris value
    Returns:
        DataFrame with the ingestion schema.
    """
    times = _history_times(n_hist)
    rows = []
    for s_i, sat in enumerate(sats):
        for i, t in enumerate(times):
            rows.append({
                "Timestamp": t, "SatelliteID": sat,
                "OrbitType": "GEO" if s_i == 0 else "MEO",
                "ClockError_ns":    clock_start + clock_slope * i + s_i,
                "EphemerisError_m": eph_start + eph_slope * i,
            })
    return pd.DataFrame(rows)


def make_truth(sats=SATS, n_steps=96, clock_slope=2.0, clock_start=100.0,
               n_hist=96, eph_slope=0.0, eph_start=5.0) -> pd.DataFrame:
    """
    Build day-8 truth continuing the same exact lines make_input() produced.

    Args:
        see make_input; n_hist fixes where the history ended so the truth
        continues it without a break.
    Returns:
        DataFrame with the ingestion schema, covering the forecast epochs.
    """
    rows = []
    for s_i, sat in enumerate(sats):
        for k, t in enumerate(_forecast_times(n_steps), start=1):
            rows.append({
                "Timestamp": t, "SatelliteID": sat,
                "OrbitType": "GEO" if s_i == 0 else "MEO",
                "ClockError_ns":    clock_start + clock_slope * (n_hist - 1 + k) + s_i,
                "EphemerisError_m": eph_start + eph_slope * (n_hist - 1 + k),
            })
    return pd.DataFrame(rows)


def make_submission(truth: pd.DataFrame, clock_offset=0.0, eph_offset=0.0) -> pd.DataFrame:
    """
    Build a submission frame from truth, displaced by a known constant.

    Args:
        truth:        frame from make_truth()
        clock_offset: constant added to every clock prediction
        eph_offset:   constant added to every ephemeris prediction
    Returns:
        DataFrame with the submission schema written by run_pipeline.py.
    """
    rows = []
    for sat, grp in truth.groupby("SatelliteID"):
        grp = grp.sort_values("Timestamp")
        for k, (_, r) in enumerate(grp.iterrows(), start=1):
            rows.append({
                "SatelliteID": sat, "Timestamp": r["Timestamp"],
                "PredictionStep": k, "HorizonMinutes": k * 15,
                "ClockError_ns_predicted":    r["ClockError_ns"] + clock_offset,
                "ClockError_ns_sigma":        1.0,
                "ClockError_ns_lower95":      r["ClockError_ns"] + clock_offset - 2,
                "ClockError_ns_upper95":      r["ClockError_ns"] + clock_offset + 2,
                "EphemerisError_m_predicted": r["EphemerisError_m"] + eph_offset,
                "EphemerisError_m_sigma":     0.1,
                "EphemerisError_m_lower95":   r["EphemerisError_m"] + eph_offset - 0.2,
                "EphemerisError_m_upper95":   r["EphemerisError_m"] + eph_offset + 0.2,
            })
    return pd.DataFrame(rows)


def write_all(tmp_path, sub, truth, inp) -> dict:
    """Write the three frames to CSVs under tmp_path and return their paths."""
    paths = {"submission": tmp_path / "submission.csv",
             "truth":      tmp_path / "holdout.csv",
             "input":      tmp_path / "input.csv"}
    sub.to_csv(paths["submission"], index=False)
    truth.to_csv(paths["truth"], index=False)
    inp.to_csv(paths["input"], index=False)
    return {k: str(v) for k, v in paths.items()}


# ── horizon keys ────────────────────────────────────────────────────────────

def test_horizon_keys_parse_to_the_steps_the_metric_uses():
    """Key names must map to the same step counts compute_rmse_horizons uses."""
    assert horizon_steps("15min") == 1
    assert horizon_steps("30min") == 2
    assert horizon_steps("1hr") == 4
    assert horizon_steps("2hr") == 8
    assert horizon_steps("12hr") == 48
    assert horizon_steps("24hr") == 96
    assert horizon_steps("at_24hr") == 96
    assert horizon_steps("nonsense") is None


def test_order_puts_cumulative_before_terminal():
    """Report order is cumulative first, then terminal, each ascending."""
    keys = compute_rmse_horizons(np.zeros(96), np.zeros(96)).keys()
    ordered = order_keys(keys)
    cumulative = [k for k in ordered if not k.startswith("at_")]
    assert ordered[:len(cumulative)] == cumulative
    assert [horizon_steps(k) for k in cumulative] == sorted(
        horizon_steps(k) for k in cumulative)


# ── accuracy ────────────────────────────────────────────────────────────────

def test_perfect_prediction_scores_zero_at_every_horizon():
    """A forecast equal to truth must score exactly 0.0, cumulative and terminal."""
    truth = make_truth()
    result = score(make_submission(truth), truth, make_input())
    assert result["horizon_keys"], "no horizons were scored"
    for col in ("ClockError_ns", "EphemerisError_m"):
        data = result["columns"][col]
        for key in result["horizon_keys"]:
            assert data["overall"]["model"][key] == pytest.approx(0.0, abs=1e-12)
            for sat, entry in data["per_sat"].items():
                assert entry["model"][key] == pytest.approx(0.0, abs=1e-12), (col, sat, key)


def test_constant_offset_produces_exactly_that_rmse():
    """A prediction displaced by a constant scores that constant at every horizon."""
    truth = make_truth()
    sub = make_submission(truth, clock_offset=3.0, eph_offset=-0.25)
    result = score(sub, truth, make_input())
    for col, expected in (("ClockError_ns", 3.0), ("EphemerisError_m", 0.25)):
        data = result["columns"][col]
        for key in result["horizon_keys"]:
            assert data["overall"]["model"][key] == pytest.approx(expected, abs=1e-9)
            for entry in data["per_sat"].values():
                assert entry["model"][key] == pytest.approx(expected, abs=1e-9)


def test_row_order_is_never_assumed():
    """Shuffling truth and submission rows must not change any figure."""
    truth = make_truth()
    sub = make_submission(truth, clock_offset=1.5)
    ordered = score(sub, truth, make_input())
    shuffled = score(sub.sample(frac=1.0, random_state=1),
                     truth.sample(frac=1.0, random_state=2), make_input())
    for col in ("ClockError_ns", "EphemerisError_m"):
        for key in ordered["horizon_keys"]:
            assert (shuffled["columns"][col]["overall"]["model"][key]
                    == pytest.approx(ordered["columns"][col]["overall"]["model"][key]))


def test_pooling_reproduces_the_metric_for_a_single_satellite():
    """
    Pooled figures must agree with compute_rmse_horizons on one satellite.

    This is the guard against the pooled path drifting away from the imported
    metric if the horizon set there ever changes.
    """
    truth = make_truth(sats=["S01"])
    sub = make_submission(truth, clock_offset=0.7)
    result = score(sub, truth, make_input(sats=["S01"]))
    y_true = truth.sort_values("Timestamp")["ClockError_ns"].to_numpy(float)
    direct = compute_rmse_horizons(y_true, y_true + 0.7)
    for key, expected in direct.items():
        got = result["columns"]["ClockError_ns"]["overall"]["model"][key]
        assert got == pytest.approx(expected, abs=1e-12), key


def test_short_horizon_leaves_far_terminal_keys_nan():
    """With fewer steps than a horizon, its terminal figure is NaN, not invented."""
    truth = make_truth(n_steps=8)
    result = score(make_submission(truth), truth, make_input())
    overall = result["columns"]["ClockError_ns"]["overall"]["model"]
    assert np.isnan(overall["at_24hr"])
    assert overall["at_2hr"] == pytest.approx(0.0, abs=1e-12)


# ── baselines ───────────────────────────────────────────────────────────────

def test_baselines_are_built_on_the_right_window():
    """
    Persistence anchors on the final observation; linear fits the last SEQ_LEN.

    The history is an exact line of slope 2 ns/step, so linear extrapolation is
    exactly right and persistence is wrong by 2k ns at step k. Both are checked
    against values computed by hand rather than by the code under test.
    """
    inp = make_input(sats=["S01"], n_hist=96, clock_slope=2.0, clock_start=100.0)
    hist = inp["ClockError_ns"].to_numpy(float)
    base = baseline_forecasts(hist, 96)
    anchor = 100.0 + 2.0 * 95
    assert np.allclose(base["persistence"], anchor)
    assert np.allclose(base["linear"], anchor + 2.0 * np.arange(1, 97))


def test_baseline_rmse_matches_hand_computation():
    """Linear scores 0.0 on a linear world; persistence scores the known drift."""
    truth = make_truth(sats=["S01"], clock_slope=2.0)
    result = score(make_submission(truth), truth, make_input(sats=["S01"], clock_slope=2.0))
    data = result["columns"]["ClockError_ns"]["per_sat"]["S01"]
    assert data["linear"]["24hr"] == pytest.approx(0.0, abs=1e-9)
    assert data["persistence"]["at_15min"] == pytest.approx(2.0)
    assert data["persistence"]["30min"] == pytest.approx(np.sqrt((4 + 16) / 2))
    assert data["persistence"]["1hr"] == pytest.approx(
        np.sqrt(np.mean((2.0 * np.arange(1, 5)) ** 2)))


def test_win_counts_reflect_the_comparison():
    """A perfect forecast beats persistence everywhere it is not tied."""
    truth = make_truth(clock_slope=2.0)
    result = score(make_submission(truth), truth, make_input(clock_slope=2.0))
    beaten, comparable = result["columns"]["ClockError_ns"]["wins"]["persistence"]["24hr"]
    assert comparable == len(SATS)
    assert beaten == len(SATS)


# ── the join ────────────────────────────────────────────────────────────────

def test_incomplete_join_raises_rather_than_scoring_a_subset():
    """Missing truth rows must stop the score, not shrink the denominator."""
    truth = make_truth()
    sub = make_submission(truth)
    partial_truth = truth.iloc[:-20]
    with pytest.raises(IncompleteJoinError) as excinfo:
        score(sub, partial_truth, make_input())
    message = str(excinfo.value)
    assert "20" in message
    assert "predicted pairs with NO truth" in message


def test_partial_join_is_scored_only_when_asked_and_says_so():
    """With allow_partial the subset is scored and every missing row is counted."""
    truth = make_truth()
    sub = make_submission(truth)
    result = score(sub, truth.iloc[:-20], make_input(), allow_partial=True)
    report = result["join"]
    assert result["partial"] is True
    assert report.pred_without_truth == 20
    assert report.truth_without_pred == 0
    assert report.n_matched == len(sub) - 20
    assert report.incomplete_sats, "the short satellite was not recorded"
    text = render_report(result, {"submission": "s", "truth": "t", "input": "i"})
    assert "PARTIAL RESULT" in text


def test_truth_without_prediction_is_counted_too():
    """Truth rows the forecast never covered are reported, not ignored."""
    truth = make_truth()
    sub = make_submission(truth)
    with pytest.raises(IncompleteJoinError) as excinfo:
        score(sub.iloc[:-5], truth, make_input())
    assert "truth pairs with NO prediction" in str(excinfo.value)


def test_missing_satellite_is_named():
    """A satellite present on one side only appears by name in the accounting."""
    truth = make_truth(sats=["S01"])
    sub = make_submission(make_truth(sats=["S01", "S02"]))
    _, report = align_predictions(sub, truth, allow_partial=True)
    assert report.sats_pred_only == ["S02"]
    assert "S02" in report.summary()


def test_duplicate_join_keys_are_refused():
    """A repeated (Timestamp, SatelliteID) pair makes an honest join impossible."""
    truth = make_truth()
    sub = make_submission(truth)
    doubled = pd.concat([truth, truth.iloc[:3]], ignore_index=True)
    with pytest.raises(IncompleteJoinError, match="duplicated"):
        align_predictions(sub, doubled)


def test_timestamp_misalignment_is_detected_not_absorbed():
    """
    Truth shifted by one step matches nothing, and that is reported as a failure.

    This is the case a positional comparison would score happily and wrongly.
    """
    truth = make_truth()
    sub = make_submission(truth)
    shifted = truth.copy()
    shifted["Timestamp"] = shifted["Timestamp"] + STEP
    with pytest.raises(IncompleteJoinError):
        score(sub, shifted, make_input())


def test_overlapping_truth_fails_the_leak_check():
    """Truth that reaches back into the input window is called out explicitly."""
    truth = make_truth()
    leaky = truth.copy()
    leaky["Timestamp"] = leaky["Timestamp"] - STEP
    sub = make_submission(leaky)
    result = score(sub, leaky, make_input())
    assert any("LEAK CHECK FAILED" in n for n in result["notes"])


# ── the command line ────────────────────────────────────────────────────────

def test_main_refuses_to_run_without_a_frozen_submission(tmp_path, capsys):
    """No submission means nothing to score, and a message that says why."""
    truth, inp = make_truth(), make_input()
    paths = write_all(tmp_path, make_submission(truth), truth, inp)
    os.remove(paths["submission"])
    code = score_day8.main(["--submission", paths["submission"],
                            "--truth", paths["truth"], "--input", paths["input"],
                            "--output", str(tmp_path / "out.txt")])
    assert code == 2
    assert "no frozen forecast" in capsys.readouterr().err.lower()


def test_main_writes_a_readable_report(tmp_path):
    """End to end on fixtures: the report exists and explains what it contains."""
    truth, inp = make_truth(), make_input()
    paths = write_all(tmp_path, make_submission(truth, clock_offset=1.0), truth, inp)
    out = tmp_path / "day8_score.txt"
    assert score_day8.main(["--submission", paths["submission"],
                            "--truth", paths["truth"], "--input", paths["input"],
                            "--output", str(out)]) == 0
    text = out.read_text()
    for expected in ("JOIN ACCOUNTING", "at_24hr", "beat linear", "PER SATELLITE",
                     "terminal figures", "S01"):
        assert expected in text, expected
    assert "PARTIAL RESULT" not in text


def test_main_exits_nonzero_on_an_incomplete_join(tmp_path, capsys):
    """The CLI surfaces the join failure rather than writing a flattering file."""
    truth, inp = make_truth(), make_input()
    paths = write_all(tmp_path, make_submission(truth), truth.iloc[:-20], inp)
    out = tmp_path / "day8_score.txt"
    assert score_day8.main(["--submission", paths["submission"],
                            "--truth", paths["truth"], "--input", paths["input"],
                            "--output", str(out)]) == 2
    assert not out.exists()
    assert "incomplete" in capsys.readouterr().err.lower()


def test_loaders_reject_a_file_with_the_wrong_columns(tmp_path):
    """A file missing a required column fails at load, not silently mid-score."""
    bad = pd.DataFrame({"Timestamp": ["2026-01-01 00:00:00"], "SatelliteID": ["S01"]})
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_truth(str(path))
    with pytest.raises(ValueError, match="missing required columns"):
        load_submission(str(path))
    with pytest.raises(ValueError, match="missing required columns"):
        load_input(str(path))
