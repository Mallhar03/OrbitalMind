"""
Tests for Iteration 8: Full Pipeline Integration
This is the final gate. ALL prior tests must also pass.

Two assertions changed here on purpose.

  * The submission is checked for the day-8 property: the forecast must sit
    beyond the end of the input record, not inside it. The old windows
    re-predicted day 7 whenever the file held 7 days.
  * Shapiro-Wilk is no longer asserted to PASS. The old post-processor
    manufactured that result from the ground truth, so the assertion was
    always satisfied and measured nothing. The pipeline now reports the real
    p-value on held-out residuals; the test checks that it is reported
    honestly, and the accuracy gate is enforced against a persistence
    baseline instead.
"""
import pytest
import os
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

PYTHON = sys.executable
DATA   = "data/synthetic/gnss_synthetic.csv"


@pytest.fixture(scope="module")
def pipeline_run():
    """Run the pipeline once for the whole module."""
    result = subprocess.run(
        [PYTHON, 'src/orbitalmind/run_pipeline.py',
         '--data', DATA, '--output', 'outputs'],
        capture_output=True, text=True, timeout=5400
    )
    return result


def test_pipeline_runs_without_error(pipeline_run):
    assert pipeline_run.returncode == 0, \
        f"Pipeline failed with error:\n{pipeline_run.stderr[-4000:]}"


def test_submission_csv_exists(pipeline_run):
    assert os.path.exists("outputs/submission.csv"), "submission.csv not generated"


def test_submission_csv_columns(pipeline_run):
    df = pd.read_csv("outputs/submission.csv")
    required = ['SatelliteID', 'PredictionStep', 'HorizonMinutes',
                'ClockError_ns_predicted', 'EphemerisError_m_predicted']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_submission_carries_a_predictive_distribution(pipeline_run):
    """The proposal promises two outputs per point, not just a point estimate."""
    df = pd.read_csv("outputs/submission.csv")
    for col in ['ClockError_ns_sigma', 'ClockError_ns_lower95',
                'ClockError_ns_upper95', 'EphemerisError_m_sigma',
                'EphemerisError_m_lower95', 'EphemerisError_m_upper95']:
        assert col in df.columns, f"Missing distribution column: {col}"
    assert (df['ClockError_ns_lower95'] <= df['ClockError_ns_upper95']).all()
    assert (df['EphemerisError_m_lower95'] <= df['EphemerisError_m_upper95']).all()


def test_uncertainty_grows_with_horizon(pipeline_run):
    """Reconstruction sums differenced steps, so spread must accumulate."""
    df = pd.read_csv("outputs/submission.csv")
    sat = df[df['SatelliteID'] == df['SatelliteID'].iloc[0]].sort_values('PredictionStep')
    sigma = sat['ClockError_ns_sigma'].values
    assert sigma[-1] >= sigma[0], "predictive sigma does not widen with horizon"


def test_submission_has_96_steps_per_satellite(pipeline_run):
    df = pd.read_csv("outputs/submission.csv")
    for sat_id in df['SatelliteID'].unique():
        count = len(df[df['SatelliteID'] == sat_id])
        assert count == 96, \
            f"Satellite {sat_id} has {count} predictions, expected 96"


def test_predictions_are_not_placeholder_zeros(pipeline_run):
    """
    The old pipeline caught every exception and wrote np.zeros(96), producing
    a complete-looking CSV from a total failure.
    """
    df = pd.read_csv("outputs/submission.csv")
    zero_rows = (df['ClockError_ns_predicted'] == 0.0).sum()
    assert zero_rows < len(df) * 0.5, \
        f"{zero_rows}/{len(df)} clock predictions are exactly zero"


def test_forecast_continues_from_the_end_of_the_record(pipeline_run):
    """
    Day 8 must extend past the input, so the first predicted value should sit
    near the last observed value, not near the start of the record.
    """
    raw = pd.read_csv(DATA)
    sub = pd.read_csv("outputs/submission.csv")
    for sat_id in sub['SatelliteID'].unique()[:3]:
        series = raw[raw['SatelliteID'] == sat_id].sort_values('Timestamp')
        last   = series['ClockError_ns'].values[-1]
        first  = series['ClockError_ns'].values[0]
        pred   = sub[(sub['SatelliteID'] == sat_id) &
                     (sub['PredictionStep'] == 1)]['ClockError_ns_predicted'].iloc[0]
        assert abs(pred - last) <= abs(pred - first) or np.isclose(last, first), \
            (f"{sat_id}: step-1 forecast {pred:.3f} is anchored nearer the "
             f"record start {first:.3f} than its end {last:.3f}")


def test_evaluation_report_exists(pipeline_run):
    assert os.path.exists("outputs/evaluation_report.txt"), \
        "evaluation_report.txt not generated"


def test_report_states_units_and_baseline(pipeline_run):
    """RMSE must be reported in ns/m against a baseline, not in filtered space."""
    with open("outputs/evaluation_report.txt") as f:
        content = f.read()
    assert "ORIGINAL units" in content, "report does not state its units"
    assert "persistence" in content, "report carries no baseline comparison"


def test_shapiro_wilk_is_reported_honestly(pipeline_run):
    """
    The p-value must be measured on held-out residuals. We assert it is
    reported and in range -- not that it passes, because manufacturing a pass
    is exactly the defect this replaced.
    """
    assert os.path.exists("outputs/shapiro_wilk_result.txt")
    with open("outputs/shapiro_wilk_result.txt") as f:
        content = f.read()
    assert "held-out" in content, "Shapiro-Wilk not stated as held-out"
    p_line = [l for l in content.splitlines() if l.startswith("p-value:")]
    assert p_line, "no p-value reported"
    p = float(p_line[0].split(":")[1])
    assert 0.0 <= p <= 1.0
    assert p != pytest.approx(0.9999, abs=1e-4), \
        "p-value pinned at 0.9999 -- the circular correction is back"


def test_qq_plot_exists(pipeline_run):
    assert os.path.exists("outputs/qq_plot.png"), "Q-Q plot not generated"


def test_all_prior_tests_still_pass():
    result = subprocess.run(
        [PYTHON, '-m', 'pytest', 'tests/', '-q',
         '--ignore=tests/test_pipeline.py'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, \
        f"Prior tests broken:\n{result.stdout[-4000:]}"
