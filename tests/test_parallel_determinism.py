"""
Parallel execution must not change results.

plan/day-02.md requires the satellite loop to be parallelised across processes
with the criterion: "Determinism must hold — same seed, same results, regardless
of worker count."

Two things had to be true for that, and both are easy to break silently:

  1. Seeds are set PER SATELLITE inside the worker. They used to be set once
     before the loop, which made each satellite's random state depend on every
     satellite processed before it — so results changed with worker count, and
     even with --max-satellites.
  2. The worker receives a pre-filtered single-satellite frame, where the serial
     path passed the whole dataset. preprocess_satellite() filters internally, so
     these must agree.

These tests are cheap. They do not train models; they check the two invariants
that make the expensive equivalence hold.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.preprocessing.pipeline import preprocess_satellite  # noqa: E402

INPUT_CSV = "data/raw/gnss_real.csv"
OUTPUTS = ("trend", "periodic", "noise", "observed", "original_cleaned")


@pytest.fixture(scope="module")
def real_df():
    if not os.path.exists(INPUT_CSV):
        pytest.skip(f"{INPUT_CSV} not present — run scripts/fetch_data.py first")
    return pd.read_csv(INPUT_CSV)


@pytest.mark.parametrize("sat", ["G01", "J07", "C06", "G19"])
@pytest.mark.parametrize("col", ["satclockerror (m)", "x_error (m)"])
def test_filtered_frame_matches_full_frame(real_df, sat, col):
    """
    The parallel worker's pre-filtered frame must preprocess identically.

    The worker passes only one satellite's rows; the serial path passed the whole
    dataset. If preprocessing ever gained cross-satellite state, this would catch
    it — and every parallel result would silently differ from the serial one.
    """
    full = preprocess_satellite(real_df, sat, col)
    filtered = preprocess_satellite(real_df[real_df.SatelliteID == sat].copy(), sat, col)

    for key in OUTPUTS:
        assert np.array_equal(
            np.asarray(full[key], dtype=float), np.asarray(filtered[key], dtype=float)
        ), f"{sat}/{col}: '{key}' differs between full and filtered frames"
    assert full["first_value"] == filtered["first_value"]


def test_worker_seeds_per_satellite_not_once_globally():
    """
    _process_satellite must seed inside itself.

    Seeding once before the loop makes satellite N's random state depend on
    satellites 1..N-1, so results change with worker count. That is precisely the
    bug this parallelisation had to avoid, and it leaves no visible symptom —
    the numbers are simply different, and plausibly so.
    """
    import inspect
    from orbitalmind import run_pipeline as rp

    src = inspect.getsource(rp._process_satellite)
    assert "np.random.seed(42)" in src, "worker does not seed numpy per satellite"
    assert "torch.manual_seed(42)" in src, "worker does not seed torch per satellite"
    assert "torch.set_num_threads(1)" in src, (
        "worker does not pin torch to one thread — N processes each spawning N "
        "threads oversubscribes the machine and runs slower than serial"
    )


def test_run_pipeline_exposes_workers():
    """--workers must exist and default to serial in the API."""
    import inspect
    from orbitalmind import run_pipeline as rp

    sig = inspect.signature(rp.run_pipeline)
    assert "workers" in sig.parameters
    assert sig.parameters["workers"].default == 1, (
        "run_pipeline(workers=...) must default to 1 so library callers are "
        "explicit; the CLI is what picks a machine-sized default"
    )


def test_results_are_collected_in_satellite_order():
    """
    Completion order must not leak into the output.

    Workers finish in arbitrary order; the code reassembles by index. If that
    ever became a plain append, submission.csv row order would vary run to run
    while every individual number stayed correct — the hardest kind of
    non-determinism to notice.
    """
    import inspect
    from orbitalmind import run_pipeline as rp

    src = inspect.getsource(rp.run_pipeline)
    assert "collected[i] for i in range(len(tasks))" in src, (
        "results are no longer reassembled in satellite order"
    )
