"""
Causal preprocessing tests: the `limit` parameter of preprocess_satellite.

None of the four preprocessing steps is causal. Outlier removal uses a CENTRED
local window, IOD correction derives its threshold from the whole series, and
EMD builds spline envelopes through every extremum, so its trend is a residue
computed from the entire record. Running all four over a full record and only
afterwards slicing out a training window therefore produces a training signal
that is a function of values lying AFTER it.

That was measured, not assumed. Perturbing only the samples after index 480 of a
real satellite's clock series and re-decomposing the whole record moves the
training-window signal by tens of percent RMS -- 56% on C08, 48% on C11, 17% on
C13 under the perturbation test_leak_exists_without_limit applies. An internal
backtest scored from such a signal has already been shown the answer.

`limit` truncates the RAW series before the first transform runs, which is the
only place truncation actually closes that path. These tests assert the two
things that matter: that the default behaviour did not move at all, and that
with a limit set the result depends on nothing after the limit.

The day-8 submission is unaffected either way -- its forecast window lies past
the end of the input record, so there is nothing after it for EMD to see -- and
that is why limit=None had to stay bit-identical.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.preprocessing.pipeline import MIN_LIMIT, preprocess_satellite  # noqa: E402
from orbitalmind.preprocessing.outlier_removal import remove_outliers_mad  # noqa: E402
from orbitalmind.preprocessing.iod_correction import correct_iod_jumps  # noqa: E402
from orbitalmind.preprocessing.differencing import single_difference  # noqa: E402
from orbitalmind.preprocessing.decomposition import decompose_signal  # noqa: E402

INPUT_CSV = "data/raw/gnss_real.csv"

# Five days of 15-minute samples. Comfortably inside every satellite's record
# (the shortest in the real file holds 651 rows) and far enough from the end
# that a perturbation after it has plenty of room to leak backwards.
LIMIT = 480

ARRAY_KEYS = ["trend", "periodic", "noise", "observed", "original_cleaned"]

# EMD reconstruction is exact to floating point; the same tolerance the S3
# decompose tests use.
RECON_TOL = 1e-14


@pytest.fixture(scope="module")
def real_df():
    if not os.path.exists(INPUT_CSV):
        pytest.skip(f"{INPUT_CSV} not present — run scripts/fetch_data.py first")
    df = pd.read_csv(INPUT_CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df


def _sat_len(df, sat_id):
    return int((df["SatelliteID"] == sat_id).sum())


def _reference_preprocess(df, sat_id, error_col):
    """
    The four steps exactly as they ran before `limit` existed.

    Written out here rather than imported so that the default path is compared
    against an independent statement of what it used to do. If someone later
    reorders the steps or slips a truncation in ahead of one of them, this
    diverges.
    """
    sat_df = df[df["SatelliteID"] == sat_id].sort_values("Timestamp").copy()
    raw_series = sat_df[error_col].reset_index(drop=True)
    timestamps = pd.to_datetime(sat_df["Timestamp"].values)

    observed = remove_outliers_mad(raw_series)
    cleaned = correct_iod_jumps(observed)
    differenced, first_value = single_difference(cleaned)
    trend, periodic, noise = decompose_signal(differenced.values.astype(float))
    return {
        "trend": trend, "periodic": periodic, "noise": noise,
        "first_value": first_value,
        "observed": observed.values.astype(float),
        "original_cleaned": cleaned.values.astype(float),
        "timestamps": timestamps[1:],
    }


def _perturb_after(df, sat_id, error_col, limit, scale=5.0, seed=0):
    """
    Replace nothing before `limit`, and scramble everything from it onwards.

    The perturbation is deliberately violent -- five times the series' own
    standard deviation -- because the claim under test is that NONE of it
    reaches the truncated output. A subtle nudge could pass by accident.
    """
    out = df.copy()
    rows = out.index[out["SatelliteID"] == sat_id]
    tail = rows[limit:]
    rng = np.random.default_rng(seed)
    spread = float(np.std(out.loc[rows, error_col]))
    out.loc[tail, error_col] = (out.loc[tail, error_col]
                                + scale * spread * rng.normal(size=len(tail)))
    return out


def _assert_same(a, b, context):
    for key in ARRAY_KEYS:
        assert np.array_equal(a[key], b[key]), f"{context}: '{key}' differs"
    assert a["first_value"] == b["first_value"], f"{context}: 'first_value' differs"
    assert np.array_equal(a["timestamps"].values, b["timestamps"].values), \
        f"{context}: 'timestamps' differ"


# ── 1. The default path must not have moved ────────────────────────────────

@pytest.mark.parametrize("sat_id", ["C08", "C11", "C13"])
@pytest.mark.parametrize("error_col", ["ClockError_ns", "EphemerisError_m"])
def test_limit_none_is_the_original_behaviour(real_df, sat_id, error_col):
    """limit=None reproduces the pre-change pipeline exactly, not approximately."""
    _assert_same(preprocess_satellite(real_df, sat_id, error_col),
                 _reference_preprocess(real_df, sat_id, error_col),
                 f"{sat_id}/{error_col} limit=None")


def test_limit_equal_to_record_length_is_a_no_op(real_df):
    """Truncating at exactly the record length keeps every sample, so nothing changes."""
    sat_id, error_col = "C08", "ClockError_ns"
    n = _sat_len(real_df, sat_id)
    _assert_same(preprocess_satellite(real_df, sat_id, error_col, limit=n),
                 preprocess_satellite(real_df, sat_id, error_col),
                 f"{sat_id} limit=n")


# ── 2. The causality test: the point of the exercise ───────────────────────

@pytest.mark.parametrize("sat_id", ["C08", "C11", "C13", "J07"])
def test_truncated_output_ignores_everything_after_the_limit(real_df, sat_id):
    """
    Scramble the raw data after `limit`, re-run with the same limit, get the
    same numbers back. Exact equality, not a tolerance: a causal transform of a
    prefix cannot see the suffix at all, so any difference is a leak.
    """
    error_col = "ClockError_ns"
    perturbed = _perturb_after(real_df, sat_id, error_col, LIMIT)
    _assert_same(preprocess_satellite(real_df, sat_id, error_col, limit=LIMIT),
                 preprocess_satellite(perturbed, sat_id, error_col, limit=LIMIT),
                 f"{sat_id} perturbed after index {LIMIT}")


def test_truncation_equals_processing_a_physically_shorter_record(real_df):
    """
    A limit is the same thing as never having received the later rows.

    This is the stronger form of the statement above: not merely that the suffix
    does not change the answer, but that the answer is the one a caller would
    get from a file that stopped at the limit.
    """
    sat_id, error_col = "C11", "EphemerisError_m"
    rows = real_df.index[real_df["SatelliteID"] == sat_id][:LIMIT]
    prefix_df = real_df.loc[rows]
    _assert_same(preprocess_satellite(real_df, sat_id, error_col, limit=LIMIT),
                 preprocess_satellite(prefix_df, sat_id, error_col),
                 f"{sat_id} prefix equivalence")


@pytest.mark.parametrize("sat_id", ["C08", "C11", "C13"])
def test_leak_exists_without_limit(real_df, sat_id):
    """
    The bug `limit` was added for, asserted as a live measurement.

    Without truncation the same perturbation moves the training-window signal
    substantially, which is what makes an untruncated backtest not out-of-sample.
    If this ever stops holding, either the decomposition became causal on its own
    or the perturbation stopped biting, and the test above stopped proving
    anything -- either way someone should look.
    """
    error_col = "ClockError_ns"
    perturbed = _perturb_after(real_df, sat_id, error_col, LIMIT)
    base = preprocess_satellite(real_df, sat_id, error_col)
    moved = preprocess_satellite(perturbed, sat_id, error_col)

    train = slice(0, LIMIT - 1)
    a = (base["trend"] + base["periodic"])[train]
    b = (moved["trend"] + moved["periodic"])[train]
    drift = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2))
    assert drift > 0.01, (
        f"{sat_id}: training window moved only {100 * drift:.3f}% RMS when the "
        f"future was scrambled — the leak this test documents may be gone"
    )


# ── 3. Shapes and internal consistency under truncation ────────────────────

@pytest.mark.parametrize("limit", [MIN_LIMIT, 97, LIMIT])
def test_lengths_and_alignment_under_truncation(real_df, limit):
    """
    Every returned series is recomputed from the truncated record, so the index
    mapping the caller anchors on survives: original-scale entry i is raw row i,
    differenced index i is raw row i+1.
    """
    sat_id, error_col = "C08", "ClockError_ns"
    pre = preprocess_satellite(real_df, sat_id, error_col, limit=limit)

    for key in ["trend", "periodic", "noise"]:
        assert len(pre[key]) == limit - 1, f"'{key}' should hold limit-1 samples"
    assert len(pre["observed"]) == limit
    assert len(pre["original_cleaned"]) == limit
    assert len(pre["timestamps"]) == limit - 1

    sat_rows = real_df[real_df["SatelliteID"] == sat_id].sort_values("Timestamp")
    expected_ts = pd.to_datetime(sat_rows["Timestamp"].values)[1:limit]
    assert np.array_equal(pre["timestamps"].values, expected_ts.values), \
        "timestamps must be the truncated raw timestamps from row 1 onwards"

    assert pre["first_value"] == pre["original_cleaned"][0], \
        "first_value must be the anchor of the TRUNCATED cleaned series"

    recon = pre["trend"] + pre["periodic"] + pre["noise"]
    err = float(np.max(np.abs(recon - np.diff(pre["original_cleaned"]))))
    assert err < RECON_TOL, f"EMD completeness broken under truncation: {err}"

    for key in ARRAY_KEYS:
        assert not np.any(np.isnan(pre[key])), f"NaN in '{key}' under truncation"


def test_truncation_actually_changes_the_result(real_df):
    """
    Guard against a limit that is quietly ignored: a truncated run must not
    equal the full run's leading slice, because the cleaning steps see a
    different record. If these ever matched, the parameter would be inert and
    the causality test above would pass for the wrong reason.
    """
    sat_id, error_col = "C08", "ClockError_ns"
    full = preprocess_satellite(real_df, sat_id, error_col)
    cut = preprocess_satellite(real_df, sat_id, error_col, limit=LIMIT)
    assert len(cut["trend"]) < len(full["trend"])
    assert not np.array_equal(cut["trend"], full["trend"][:len(cut["trend"])])


# ── 4. Degenerate limits fail loudly ───────────────────────────────────────

def test_limit_beyond_the_record_raises(real_df):
    """
    A limit past the end of the record means the caller's window arithmetic was
    computed against a length this satellite does not have. Returning the short
    series instead would satisfy the call and silently shift every downstream
    index, so it raises.
    """
    sat_id = "C08"
    n = _sat_len(real_df, sat_id)
    with pytest.raises(ValueError, match="exceeds"):
        preprocess_satellite(real_df, sat_id, "ClockError_ns", limit=n + 1)


@pytest.mark.parametrize("limit", [-1, 0, 1, 2])
def test_limit_too_small_raises(real_df, limit):
    """
    Below MIN_LIMIT there is no well-formed result to return: differencing costs
    one sample and EMD needs two, so limit=2 would otherwise reach PyEMD and die
    with 'zero-size array to reduction operation minimum'. Fail in the caller's
    vocabulary instead.
    """
    with pytest.raises(ValueError, match="too small"):
        preprocess_satellite(real_df, "C08", "ClockError_ns", limit=limit)


def test_smallest_allowed_limit_still_produces_a_valid_result(real_df):
    """MIN_LIMIT is the boundary, so it must work rather than merely not raise."""
    pre = preprocess_satellite(real_df, "C08", "ClockError_ns", limit=MIN_LIMIT)
    assert len(pre["trend"]) == MIN_LIMIT - 1
    assert len(pre["observed"]) == MIN_LIMIT
    recon = pre["trend"] + pre["periodic"] + pre["noise"]
    assert np.max(np.abs(recon - np.diff(pre["original_cleaned"]))) < RECON_TOL


@pytest.mark.parametrize("limit", [4.0, "480", True])
def test_non_integer_limit_raises_typeerror(real_df, limit):
    """
    bool is an int subclass, so limit=True would otherwise mean 'keep one
    sample' — a typo that would silently destroy a run rather than stop it.
    """
    with pytest.raises(TypeError, match="must be an int"):
        preprocess_satellite(real_df, "C08", "ClockError_ns", limit=limit)


def test_numpy_integer_limit_is_accepted(real_df):
    """Split indices arrive as numpy integers; rejecting them would be a trap."""
    a = preprocess_satellite(real_df, "C08", "ClockError_ns", limit=np.int64(LIMIT))
    b = preprocess_satellite(real_df, "C08", "ClockError_ns", limit=LIMIT)
    _assert_same(a, b, "np.int64 limit")
