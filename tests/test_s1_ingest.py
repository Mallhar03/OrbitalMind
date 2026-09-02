"""
S1 INGEST stage tests.

These lock in the three defects the S1 gate found and that were fixed in
scripts/fetch_data.py. Each of these bugs shipped once and none of them
announced itself — the row counts still looked plausible, so only an explicit
assertion catches a regression.

  1. ClockError_ns held the raw SP3 precise clock bias rather than a
     broadcast-minus-precise error (values ~1e5 ns against a 0.65 ns target).
  2. The source product was GPS-only, so the dataset had zero GEO/GSO
     satellites and the deck's orbit-type branching had nothing to train on.
  3. SP3 daily files repeat the midnight epoch at both ends, so concatenating
     per-day frames produced a duplicate (Timestamp, SatelliteID) row at every
     day boundary, with the two copies disagreeing by up to 6 ns.

Tests that need the pulled dataset skip cleanly when it is absent, so the suite
still runs on a machine that has never fetched data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import fetch_data as F  # noqa: E402

INPUT_CSV   = "data/raw/gnss_real.csv"
HOLDOUT_CSV = "data/raw/gnss_holdout.csv"

STEP_MIN     = 15
ROWS_PER_DAY = 24 * 60 // STEP_MIN          # 96


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present — run scripts/fetch_data.py first")
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df


# ---------------------------------------------------------------- unit tests
# These need no network and no pulled data, so the invariants stay enforced
# even on a clean checkout.

def test_sort_values_is_stable_under_ties():
    """
    The dedup fix depends on pandas' multi-column sort being stable.

    keep="last" only selects the day-owning copy because equal
    (SatelliteID, Timestamp) pairs retain their original concatenation order,
    oldest day first. If a future pandas made sorting unstable, the fix would
    silently start picking arbitrary rows with no visible symptom.
    """
    df = pd.DataFrame({
        "SatelliteID": ["G01"] * 4,
        "Timestamp":   pd.to_datetime(["2026-01-02"] * 4),
        "order":       [0, 1, 2, 3],
    })
    out = df.sort_values(["SatelliteID", "Timestamp"])
    assert list(out["order"]) == [0, 1, 2, 3]


def test_dedup_keeps_the_day_owning_copy():
    """
    At a day boundary the later frame wins, which is the day that owns the epoch.

    Frames are concatenated oldest-first, so for the shared midnight epoch the
    second copy comes from the day starting at it, not from the previous day's
    trailing 24:00 endpoint. Keeping that copy means each day's series comes
    from a single SP3 product and a single nav file throughout.
    """
    boundary = pd.Timestamp("2026-08-22 00:00:00")
    day_prev = pd.DataFrame({"Timestamp": [boundary], "SatelliteID": ["G01"],
                             "OrbitType": ["MEO"], "ClockError_ns": [1.0],
                             "EphemerisError_m": [0.1]})
    day_next = pd.DataFrame({"Timestamp": [boundary], "SatelliteID": ["G01"],
                             "OrbitType": ["MEO"], "ClockError_ns": [2.0],
                             "EphemerisError_m": [0.2]})

    combined = (pd.concat([day_prev, day_next], ignore_index=True)
                  .sort_values(["SatelliteID", "Timestamp"])
                  .reset_index(drop=True))
    deduped = combined.drop_duplicates(subset=["Timestamp", "SatelliteID"],
                                       keep="last")

    assert len(deduped) == 1
    assert deduped.iloc[0]["ClockError_ns"] == 2.0, "kept the previous day's copy"


def test_classify_orbit_uses_geometry_not_prn():
    """Geosynchronous radius means GEO; MEO radius means MEO, whatever the PRN."""
    geo = np.tile(np.array([[42164.0, 0.0, 0.0]]), (10, 1))
    meo = np.tile(np.array([[26560.0, 0.0, 0.0]]), (10, 1))
    assert F.classify_orbit(geo) == "GEO"
    assert F.classify_orbit(meo) == "MEO"

    # An inclined geosynchronous satellite is still GEO for the deck's purposes:
    # it carries the same 24-hour periodicity the GEO branch models.
    igso = np.column_stack([
        np.full(10, 30000.0), np.zeros(10), np.full(10, 29000.0),
    ])
    assert F.classify_orbit(igso) == "GEO"


def test_broadcast_clock_excludes_relativistic_term():
    """
    The clock correction must be the bare polynomial.

    The relativistic term F*e*sqrt(A)*sin(Ek) cancels between the broadcast and
    IGS precise clocks. Adding it injects an eccentricity-dependent signal that
    measured +/-400 ns on Galileo's eccentric E14/E18, against a 0.65 ns target.
    """
    from datetime import datetime

    toc   = datetime(2026, 8, 28, 12, 0, 0)
    epoch = datetime(2026, 8, 28, 12, 10, 0)
    dt    = (epoch - toc).total_seconds()

    params = {
        "SVclockBias": 1e-4, "SVclockDrift": 2e-12, "SVclockDriftRate": 0.0,
        # High eccentricity: if the relativistic term leaked in, it would show.
        "Eccentricity": 0.16, "sqrtA": 5289.0, "Toe": 475200.0,
        "M0": -1.7, "DeltaN": 6e-9,
    }
    expected = params["SVclockBias"] + params["SVclockDrift"] * dt
    got = F.broadcast_clock_s(params, toc, epoch, "E")
    assert got == pytest.approx(expected, abs=1e-15)


# ------------------------------------------------------- shipped dataset tests

def test_no_duplicate_keys():
    """One row per satellite per epoch, in both files."""
    for path in (INPUT_CSV, HOLDOUT_CSV):
        df = _load(path)
        dups = df.duplicated(subset=["Timestamp", "SatelliteID"]).sum()
        assert dups == 0, f"{path} has {dups} duplicate (Timestamp, SatelliteID) rows"


def test_schema_matches_deck():
    """Exact columns the solution deck specifies, and no NaN."""
    for path in (INPUT_CSV, HOLDOUT_CSV):
        df = _load(path)
        assert list(df.columns) == F.REQUIRED_COLS
        assert df.isnull().sum().sum() == 0


def test_both_orbit_types_present():
    """
    The deck sells GEO/GSO and MEO branching, so both must be in the data.

    A GPS-only product yields 100% MEO and silently starves the GEO branch.
    """
    df = _load(INPUT_CSV)
    types = set(df["OrbitType"].unique())
    assert types == {"GEO", "MEO"}, f"expected both orbit types, got {types}"
    assert df[df.OrbitType == "GEO"]["SatelliteID"].nunique() >= 1


def test_clock_error_is_an_error_not_a_raw_bias():
    """
    Guards the original P0: ClockError_ns must not be the raw SP3 clock bias.

    A raw bias is hundreds of microseconds and shows a huge per-satellite mean
    against a tiny per-satellite spread. A real broadcast clock error is orders
    of magnitude smaller.
    """
    df = _load(INPUT_CSV)
    assert df["ClockError_ns"].abs().max() < 10_000, \
        "ClockError_ns is microsecond-scale — looks like a raw clock bias"

    stats = df.groupby("SatelliteID")["ClockError_ns"].agg(["mean", "std"])
    ratio = (stats["mean"].abs() / stats["std"].replace(0, np.nan)).max()
    assert ratio < 1000, \
        f"per-satellite mean/std ratio {ratio:.0f} indicates an uncorrected bias"


def test_ephemeris_error_is_not_clipped():
    """
    Guards the clipping defect: no pile-up on a fixed boundary.

    The old code clipped to [-5, 5] m, parking 7% of rows exactly on -5.0.
    """
    df = _load(INPUT_CSV)
    for bound in (-5.0, 5.0):
        stuck = (df["EphemerisError_m"] == bound).sum()
        assert stuck < 10, f"{stuck} rows sit exactly on {bound} m — clipping is back"


def test_holdout_is_disjoint_from_input():
    """
    Day 8 is held-out truth and must never appear in the training input.

    Any shared (Timestamp, SatelliteID) is a leak that would invalidate every
    score computed against it.
    """
    a, b = _load(INPUT_CSV), _load(HOLDOUT_CSV)
    overlap = a.merge(b, on=["Timestamp", "SatelliteID"], how="inner")
    assert len(overlap) == 0, f"{len(overlap)} rows leak between input and holdout"

    assert a["Timestamp"].max() < b["Timestamp"].min(), \
        "input must end strictly before the holdout day begins"
    assert b["Timestamp"].dt.date.nunique() == 1, "holdout must be exactly one day"


def test_row_counts_match_the_deck_window():
    """
    7 days in, 1 day held out, on a 15-minute grid.

    The deck states ~672 rows per satellite (7 x 96). Satellites with real data
    gaps fall short, which is honest; none may exceed the window.
    """
    a, b = _load(INPUT_CSV), _load(HOLDOUT_CSV)

    assert a["Timestamp"].dt.date.nunique() == 7
    input_rows = a.groupby("SatelliteID").size()
    assert input_rows.max() <= 7 * ROWS_PER_DAY
    assert input_rows.mode()[0] == 7 * ROWS_PER_DAY

    holdout_rows = b.groupby("SatelliteID").size()
    assert holdout_rows.max() <= ROWS_PER_DAY
    assert holdout_rows.mode()[0] == ROWS_PER_DAY
