"""
S3 DECOMPOSE stage tests.

EMD's completeness property means trend + periodic + noise reproduces the input
to floating-point precision. That guarantee is the reason Decision 001 chose EMD
over EWT, and it is what makes the decomposition safe to build on — so it is
worth asserting at the precision it actually holds to.

tests/test_preprocessing.py checks reconstruction with a tolerance of 1e-4,
roughly twelve orders of magnitude looser than the ~1e-16 the implementation
achieves, and only on synthetic data. A regression that degraded reconstruction
by eleven orders of magnitude would still pass it. These tests close that gap and
cover the degenerate IMF-count branches, which no test previously exercised.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.preprocessing.decomposition import decompose_signal  # noqa: E402
from orbitalmind.preprocessing.pipeline import preprocess_satellite  # noqa: E402

INPUT_CSV = "data/raw/gnss_real.csv"

# EMD reconstruction is exact to floating point. Measured across 40 real series
# it lands between 2.2e-16 and 8.9e-16, so this asserts the real guarantee with
# a little headroom rather than a tolerance that would hide a regression.
RECON_TOL = 1e-14


@pytest.fixture(scope="module")
def real_df():
    if not os.path.exists(INPUT_CSV):
        pytest.skip(f"{INPUT_CSV} not present — run scripts/fetch_data.py first")
    df = pd.read_csv(INPUT_CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df


# ------------------------------------------------------------- reconstruction

@pytest.mark.parametrize("sat", ["G01", "J07", "C06", "G19"])
@pytest.mark.parametrize("col", ["ClockError_ns", "EphemerisError_m"])
def test_reconstruction_is_exact_on_real_data(real_df, sat, col):
    """
    trend + periodic + noise must reproduce the differenced input exactly.

    G19 is included deliberately: it is the ragged satellite (651 rows) and the
    most likely to hit an edge case in EMD.
    """
    result = preprocess_satellite(real_df, sat, col)
    recon = result["trend"] + result["periodic"] + result["noise"]
    differenced = np.diff(np.asarray(result["original_cleaned"], dtype=float))

    assert len(recon) == len(differenced)
    err = float(np.max(np.abs(recon - differenced)))
    assert err < RECON_TOL, f"{sat}/{col} reconstruction error {err:.2e}"


def test_reconstruction_tolerance_is_meaningful():
    """
    Guard the guard: the tolerance must be tight enough to catch a real break.

    A decomposition that lost a part per million would sail through a 1e-4
    tolerance, which is what the pre-existing test used.
    """
    rng = np.random.default_rng(42)
    signal = np.cumsum(rng.normal(0, 1, 400))
    trend, periodic, noise = decompose_signal(signal)

    exact = float(np.max(np.abs((trend + periodic + noise) - signal)))
    assert exact < RECON_TOL

    # A corruption a thousand times smaller than the old tolerance must fail.
    corrupted = float(np.max(np.abs((trend + periodic + noise + 1e-7) - signal)))
    assert corrupted > RECON_TOL


# -------------------------------------------------------- degenerate branches

@pytest.mark.parametrize("signal,label", [
    (np.zeros(200), "all zeros"),
    (np.full(200, 3.7), "constant"),
    (np.arange(200, dtype=float) * 0.5, "pure ramp"),
    (np.sin(2 * np.pi * np.arange(200) / 96), "pure sine"),
    (np.array([1.0, 2.0, 3.0, 4.0]), "4 samples"),
])
def test_degenerate_inputs_still_reconstruct(signal, label):
    """
    The 0/1/2-IMF fallback branches must stay exact.

    These are unreachable on the current dataset — every real series yields 6-9
    IMFs — so nothing else would notice if one of them broke.
    """
    trend, periodic, noise = decompose_signal(signal)
    assert len(trend) == len(periodic) == len(noise) == len(signal)
    err = float(np.max(np.abs((trend + periodic + noise) - signal)))
    assert err < RECON_TOL, f"{label}: reconstruction error {err:.2e}"


def test_components_are_finite_on_real_data(real_df):
    """No NaN or inf may reach S4 through any of the three layers."""
    for sat in ("G01", "J07", "G19"):
        result = preprocess_satellite(real_df, sat, "ClockError_ns")
        for name in ("trend", "periodic", "noise"):
            arr = np.asarray(result[name], dtype=float)
            assert np.all(np.isfinite(arr)), f"{sat} {name} contains non-finite values"


def test_decomposition_is_deterministic(real_df):
    """
    Fixed seeds mean the same input yields the same decomposition (C-31).

    Without this, every run reshuffles what the models train on.
    """
    a = preprocess_satellite(real_df, "G01", "ClockError_ns")
    b = preprocess_satellite(real_df, "G01", "ClockError_ns")
    for name in ("trend", "periodic", "noise"):
        assert np.allclose(a[name], b[name], atol=0, rtol=0), f"{name} not reproducible"
