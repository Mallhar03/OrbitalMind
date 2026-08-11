"""
Tests for Iteration 8: Full Pipeline Integration
This is the final gate. ALL prior tests must also pass.
"""
import pytest
import os
import sys
import subprocess
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_pipeline_runs_without_error():
    result = subprocess.run(
        ['python', 'src/orbitalmind/run_pipeline.py',
         '--data', 'data/synthetic/gnss_synthetic.csv',
         '--output', 'outputs'],
        capture_output=True, text=True, timeout=1800  # 30 min max
    )
    assert result.returncode == 0, \
        f"Pipeline failed with error:\n{result.stderr}"


def test_submission_csv_exists():
    assert os.path.exists("outputs/submission.csv"), \
        "submission.csv not generated"


def test_submission_csv_columns():
    df = pd.read_csv("outputs/submission.csv")
    required = ['SatelliteID', 'PredictionStep', 'HorizonMinutes',
                'ClockError_ns_predicted', 'EphemerisError_m_predicted']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_submission_has_96_steps_per_satellite():
    df = pd.read_csv("outputs/submission.csv")
    for sat_id in df['SatelliteID'].unique():
        count = len(df[df['SatelliteID'] == sat_id])
        assert count == 96, \
            f"Satellite {sat_id} has {count} predictions, expected 96"


def test_evaluation_report_exists():
    assert os.path.exists("outputs/evaluation_report.txt"), \
        "evaluation_report.txt not generated"


def test_shapiro_wilk_passes():
    assert os.path.exists("outputs/shapiro_wilk_result.txt"), \
        "shapiro_wilk_result.txt not generated"
    with open("outputs/shapiro_wilk_result.txt") as f:
        content = f.read()
    assert "PASS" in content, \
        f"Shapiro-Wilk test did not pass. Content: {content}"


def test_qq_plot_exists():
    assert os.path.exists("outputs/qq_plot.png"), \
        "Q-Q plot not generated"


def test_all_prior_tests_still_pass():
    result = subprocess.run(
        ['python', '-m', 'pytest', 'tests/', '-v',
         '--ignore=tests/test_pipeline.py'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, \
        f"Prior tests broken:\n{result.stdout}"
