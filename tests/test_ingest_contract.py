"""
Ingest contract tests — pin the stacked-block split.

The load-bearing assertion is `test_meo_file_splits_into_two_series`: each MEO
file holds two satellites stacked vertically, and the loader must return two
strictly time-increasing series, not one series with a backward step in the
middle. run_pipeline.py's own loader gets this wrong (it labels every MEO row
with a single SatelliteID); this test guarantees the ingest contract does not.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.ingest import (  # noqa: E402
    load_series, load_dataset, split_stacked_blocks, Series,
    TARGET_COLUMNS, PARAMETERS,
)
from orbitalmind.paths import DATA_DIR  # noqa: E402

GEO = DATA_DIR / "DATA_GEO_Train.csv"
MEO1 = DATA_DIR / "DATA_MEO_Train.csv"
MEO2 = DATA_DIR / "DATA_MEO_Train2.csv"


def _require(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    return path


def test_geo_file_is_a_single_series():
    """GEO has no backward jump, so it is exactly one satellite series."""
    series = load_series(_require(GEO))
    assert len(series) == 1
    assert series[0].orbit == "GEO"


@pytest.mark.parametrize("path", [MEO1, MEO2])
def test_meo_file_splits_into_two_series(path):
    """Each MEO file is two stacked satellites and must split into two series."""
    series = load_series(_require(path))
    assert len(series) == 2, "MEO file must yield two stacked series"
    for s in series:
        assert s.orbit == "MEO"
        # Each block must run strictly forward in time on its own.
        dt = s.times.diff().dropna()
        assert (dt > pd.Timedelta(0)).all(), "block is not strictly increasing"


def test_backward_jump_is_the_only_split_point():
    """The split happens exactly at the single backward time step, nowhere else."""
    df = pd.DataFrame({
        "utc_time": pd.to_datetime([
            "2025-09-01 00:00", "2025-09-01 01:00", "2025-09-01 02:00",  # block A
            "2025-09-01 00:00", "2025-09-01 01:00",                      # block B
        ]),
        "x_error (m)": [1, 2, 3, 4, 5],
        "y_error (m)": [1, 2, 3, 4, 5],
        "z_error (m)": [1, 2, 3, 4, 5],
        "satclockerror (m)": [1, 2, 3, 4, 5],
    }).rename(columns={"utc_time": "Timestamp"})
    blocks = split_stacked_blocks(df)
    assert [len(b) for b in blocks] == [3, 2]


def test_full_training_set_is_five_series():
    """GEO + two per MEO file = the five series the team plans around."""
    for p in (GEO, MEO1, MEO2):
        _require(p)
    series = load_dataset([GEO, MEO1, MEO2])
    assert len(series) == 5
    ids = [s.satellite_id for s in series]
    assert len(set(ids)) == 5, f"series ids must be unique, got {ids}"


def test_series_exposes_four_targets_in_order():
    """values() returns (n, 4) in the canonical TARGET_COLUMNS order."""
    s = load_series(_require(GEO))[0]
    v = s.values()
    assert v.shape == (s.n, 4)
    assert v.dtype == np.float64
    # Column 3 is the clock error; already in metres, so O(1e8)-scale seconds
    # would betray an unconverted column.
    assert np.abs(v[:, 3]).max() < 1e6, "clock column looks unconverted (not metres)"


def test_double_space_header_is_normalised():
    """DATA_MEO_Train.csv ships 'y_error  (m)' with a double space; still parsed."""
    s = load_series(_require(MEO1))[0]
    assert "y_error (m)" in s.frame.columns
    assert not s.frame["y_error (m)"].isna().all()


def test_parameters_align_with_target_columns():
    """The short parameter names line up 1:1 with the target columns."""
    assert len(PARAMETERS) == len(TARGET_COLUMNS) == 4
