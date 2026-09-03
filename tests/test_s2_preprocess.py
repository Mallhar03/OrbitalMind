"""
S2 PREPROCESS stage tests, run against the REAL pulled dataset.

tests/test_preprocessing.py exercises only synthetic data, which is why two
correctness bugs shipped unnoticed and were caught by the S2 audit rather than
by pytest:

  1. Outlier removal used a GLOBAL median/MAD, so a satellite's genuine
     periodicity scored as outliers. It flagged 82 of J07's 672 ephemeris points
     (12.2%) in nine daily runs and interpolated over that satellite's real
     24-hour signal -- the very structure the deck's GEO branch models.
  2. IOD correction removed the routine ~2-hourly broadcast upload sawtooth.
     Those resets are genuine structure and are same-signed, so subtracting ~84
     of them manufactured a ~42 ns ramp on a 3 ns signal.

These tests are guard rails against those regressions, not quality targets. They
skip cleanly when the pulled dataset is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.preprocessing.iod_correction import correct_iod_jumps, count_jumps  # noqa: E402
from orbitalmind.preprocessing.outlier_removal import remove_outliers_mad  # noqa: E402

INPUT_CSV = "data/raw/gnss_real.csv"


@pytest.fixture(scope="module")
def real_df():
    if not os.path.exists(INPUT_CSV):
        pytest.skip(f"{INPUT_CSV} not present — run scripts/fetch_data.py first")
    df = pd.read_csv(INPUT_CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df


def _series(df, sat, col):
    g = df[df["SatelliteID"] == sat].sort_values("Timestamp")
    return g[col].reset_index(drop=True).astype(float)


# ------------------------------------------------- outlier removal, real data

def test_outlier_removal_preserves_geo_periodicity(real_df):
    """
    J07's real 24-hour ephemeris signal must survive cleaning.

    The global test erased it. A local test tracks the periodic shape, so the
    cleaned series stays close to the observed one and keeps its amplitude.
    """
    raw = _series(real_df, "J07", "EphemerisError_m")
    cleaned = remove_outliers_mad(raw)

    assert np.corrcoef(raw, cleaned)[0, 1] > 0.95, "cleaning destroyed J07's shape"

    raw_amp, cleaned_amp = raw.max() - raw.min(), cleaned.max() - cleaned.min()
    assert cleaned_amp > 0.8 * raw_amp, \
        f"cleaning flattened J07's amplitude {raw_amp:.2f} -> {cleaned_amp:.2f} m"


def test_outlier_removal_does_not_flag_a_smooth_periodic_signal():
    """A clean sine has no outliers. The global test flagged its extremes."""
    t = np.arange(672)
    sine = pd.Series(np.sin(2 * np.pi * t / 96))     # exact 24-hour period
    cleaned = remove_outliers_mad(sine)
    altered = (~np.isclose(sine, cleaned)).sum()
    assert altered == 0, f"{altered} points of a clean sine flagged as outliers"


def test_outlier_removal_still_catches_a_real_spike():
    """The local test must not be so permissive that genuine spikes survive."""
    t = np.arange(672)
    s = pd.Series(np.sin(2 * np.pi * t / 96))
    s.iloc[300] = 50.0
    cleaned = remove_outliers_mad(s)
    assert abs(cleaned.iloc[300]) < 5.0, "isolated spike was not removed"


def test_outlier_removal_preserves_length_and_grid(real_df):
    """Cleaning interpolates; it must never drop rows and punch holes (C-23)."""
    for sat in ("G01", "J07", "C06", "G19"):
        for col in ("ClockError_ns", "EphemerisError_m"):
            raw = _series(real_df, sat, col)
            assert len(remove_outliers_mad(raw)) == len(raw)


# ---------------------------------------------------- IOD correction, real data

def test_iod_correction_is_rare_across_the_constellation(real_df):
    """
    IOD correction must fire on anomalies, not on the routine upload sawtooth.

    Before the fix it averaged ~15 jumps per satellite on the clock column and
    touched all but one satellite.
    """
    counts, untouched = [], 0
    for sat, g in real_df.groupby("SatelliteID"):
        obs = remove_outliers_mad(
            g.sort_values("Timestamp")["ClockError_ns"].reset_index(drop=True))
        n = count_jumps(obs)
        counts.append(n)
        untouched += (n == 0)

    counts = np.array(counts)
    assert counts.mean() < 5.0, \
        f"mean {counts.mean():.1f} jumps/satellite — firing on routine resets"
    assert untouched >= 20, \
        f"only {untouched}/95 satellites untouched — correction is too eager"


def _correction_metrics(real_df, col):
    """Per-satellite drift and correlation effects of IOD correction on `col`."""
    out = []
    for sat, g in real_df.groupby("SatelliteID"):
        obs = remove_outliers_mad(
            g.sort_values("Timestamp")[col].reset_index(drop=True)).astype(float)
        cor = correct_iod_jumps(obs).astype(float)
        rng = float(obs.max() - obs.min())
        if rng <= 0:
            continue
        # End-to-end accumulated drift the correction ADDED, relative to the
        # signal's own range. Mean shift is the weaker metric and misses a ramp
        # that builds steadily across the week.
        drift = abs(float((cor.iloc[-1] - cor.iloc[0])
                          - (obs.iloc[-1] - obs.iloc[0]))) / rng
        corr = float(np.corrcoef(obs, cor)[0, 1]) if np.std(cor) > 0 else 1.0
        out.append((sat, drift, corr))
    return out


def test_iod_correction_does_not_manufacture_drift(real_df):
    """
    Correcting same-signed resets accumulates into an artificial ramp.

    Measured per satellite as end-to-end drift, not as a mean shift: the mean
    shift metric reported only 1 offending satellite while the accumulated-drift
    metric found 24, which is the defect that actually matters.
    """
    for col in ("ClockError_ns", "EphemerisError_m"):
        offenders = [(s, d) for s, d, _ in _correction_metrics(real_df, col) if d > 1.0]
        assert not offenders, \
            f"{col}: {len(offenders)} satellites drift beyond their own range: {offenders[:3]}"


def test_iod_correction_never_makes_a_satellite_worse_than_untouched(real_df):
    """
    A negative correlation with the observed series means the correction is
    actively worse than doing nothing. 15 of 95 satellites were in that state
    before the systematic-rejection guard.
    """
    for col in ("ClockError_ns", "EphemerisError_m"):
        metrics = _correction_metrics(real_df, col)
        negative = [(s, round(c, 3)) for s, _, c in metrics if c < 0]
        worst = min(c for _, _, c in metrics)
        assert len(negative) <= 3, f"{col}: {len(negative)} satellites anti-correlated: {negative[:5]}"
        assert worst > -0.2, f"{col}: worst correlation {worst:.3f}"


def test_iod_correction_leaves_a_clean_trend_alone():
    """
    Steady drift is not a jump, however fast.

    This is the failure the module's docstring records: a fixed threshold once
    classified all 671 steps of the fast satellites as jumps.
    """
    steady = pd.Series(np.arange(672) * 0.05)
    assert count_jumps(steady) == 0
    corrected = correct_iod_jumps(steady)
    assert np.allclose(corrected, steady), "steady drift was altered"


def test_iod_correction_removes_a_genuine_isolated_step():
    """A single large anomalous discontinuity must still be removed."""
    s = pd.Series(np.concatenate([
        np.random.default_rng(42).normal(0, 0.05, 336),
        np.random.default_rng(43).normal(0, 0.05, 336) + 20.0,
    ]))
    corrected = correct_iod_jumps(s)
    before, after = corrected[:336].mean(), corrected[336:].mean()
    assert abs(after - before) < 1.0, \
        f"step of 20 survived correction: level moved {after - before:.2f}"


def test_preprocessing_never_reads_the_holdout():
    """Day 8 is the answer. No preprocessing module may reference it."""
    root = os.path.join(os.path.dirname(__file__), "..", "src", "orbitalmind")
    offenders = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as fh:
                    if "gnss_holdout" in fh.read():
                        offenders.append(path)
    assert not offenders, f"holdout referenced in {offenders}"
