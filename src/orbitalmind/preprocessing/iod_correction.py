"""
IOD (Issue of Data) jump correction for GNSS satellite error time series.

An IOD upload shows up as a step discontinuity in the broadcast-minus-precise
error. Detecting it means asking whether a step is anomalous *for this
satellite*, which a fixed threshold cannot do: GPS clock drift rates vary by
orders of magnitude across the constellation.

The original fixed 2.0 ns threshold classified every single step of the fast
satellites as a jump -- 671 of 671 for both G02 and G03 in the real IGS data --
and subtracting all of those offsets removed the entire trend, leaving the
models nothing to predict. The threshold is now derived from the robust spread
of the satellite's own first differences.

Two further corrections were needed once the data became a true
broadcast-minus-precise error rather than a raw clock bias:

  * Detection runs on the DETRENDED step. A satellite under steady drift has a
    non-zero median step, and testing the raw step magnitude counted ordinary
    drift towards the jump criterion.
  * Correction subtracts only the ANOMALOUS EXCESS, not the whole step. An IOD
    upload superimposes a discontinuity on top of the drift the satellite was
    already undergoing, so removing the entire step also removes that drift.
    Subtracting whole steps accumulated: measured on real 2026-08 data it
    shifted J02 by -25.27 ns across a signal whose own range was 5.39 ns, and
    reversed the correlation between the observed and corrected series.
"""
import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826   # scales MAD to a standard-deviation equivalent
JUMP_SIGMAS  = 8.0      # a jump must exceed this many robust sigmas
MIN_SPREAD   = 1e-9

# A genuine IOD upload changes the level of the series and that change PERSISTS.
# Ordinary dynamics produce large single differences without a lasting shift.
# Broadcast uploads are roughly two hours apart, so a one-hour window either side
# (4 samples at 15-minute cadence) stays inside a single upload interval and
# cannot straddle two of them.
LEVEL_WINDOW  = 4
LEVEL_SIGMAS  = 3.0

# Routine broadcast uploads reset the error every ~2 hours, and those resets are
# genuine structure the models must learn, not defects to erase. They are also
# same-signed, so removing all of them accumulates: measured on real 2026-08
# data, ~84 routine resets of ~0.5 ns manufactured a ~42 ns ramp on a 3 ns
# signal, driving end-to-end drift from 0.15 to 1.37 of the series' own range.
# An IOD ANOMALY is therefore a discontinuity unlike the satellite's own routine
# discontinuities -- that is what this stage removes. The routine sawtooth is
# left for single-differencing, which handles level shifts without accumulating.
ANOMALY_SIGMAS = 3.0
MIN_ROUTINE_JUMPS = 4      # below this there is no routine population to compare against

# A count threshold alone leaves a coverage gap: a satellite with only three
# resets, spaced exactly 96 samples (24 h) apart and all the same sign, is
# plainly showing routine behaviour, yet too few candidates exist to form a
# population to compare against. Regularity is therefore tested directly, and
# applies at any candidate count. Measured on real 2026-08 data, the count-only
# gate left 46 of 95 satellites unfiltered and drove 15 of them to a NEGATIVE
# correlation with the observed series -- worse than no correction at all.
ROUTINE_GAP_TOLERANCE = 0.15   # relative spread of spacings that still counts as regular
ROUTINE_MAG_TOLERANCE = 0.50   # relative spread of magnitudes that still counts as uniform

# Genuine anomalies are one-offs and have no preferred direction. If the removals
# sum to a net shift comparable to the signal's own range, they are not
# anomalies at all -- they are the satellite's systematic behaviour, and
# subtracting them manufactures a ramp that was never in the data. Removing a
# broadcast upload step must not change the multi-day trend, because that trend
# is physical (clock ageing).
MAX_NET_SHIFT_FRACTION = 1.0


def _jump_threshold(diffs: pd.Series, threshold_ns: float | None) -> float:
    """
    Choose the discontinuity threshold for one satellite.

    Args:
        diffs:        first differences of the series
        threshold_ns: explicit override, or None to derive one
    Returns:
        Absolute step size above which a difference counts as an IOD jump.
    """
    if threshold_ns is not None:
        return float(threshold_ns)

    finite = diffs.dropna().astype(float)
    if len(finite) == 0:
        return float("inf")

    centre = float(np.median(finite))
    mad    = float(np.median(np.abs(finite - centre)))
    spread = mad * MAD_TO_SIGMA

    if spread < MIN_SPREAD:
        # A perfectly regular series: only a genuine break stands out at all.
        span = float(np.max(np.abs(finite - centre)))
        return float("inf") if span < MIN_SPREAD else max(span * 0.5, MIN_SPREAD)

    # Threshold applies to the step's departure from the satellite's typical
    # step, so steady drift never counts towards the jump criterion.
    return JUMP_SIGMAS * spread


def _step_centre(diffs: pd.Series) -> float:
    """
    The satellite's typical per-step change: its ongoing drift.

    Args:
        diffs: first differences of the series
    Returns:
        Median first difference, or 0.0 when the series is too short.
    """
    finite = diffs.dropna().astype(float)
    return float(np.median(finite)) if len(finite) else 0.0


def _persistent_jump_indices(s: pd.Series, threshold_ns: float | None) -> list:
    """
    Indices of steps that are genuine, persistent level shifts.

    A candidate must clear two independent tests. It must be an anomalous step
    relative to the satellite's ordinary drift, AND the level either side of it
    must actually differ -- a real IOD upload moves the series and it stays
    moved. Requiring persistence is what separates an upload from a momentary
    excursion, and without it the correction is unstable: every false positive
    shifts all subsequent samples permanently, so frequent detections accumulate
    into an artificial staircase. Measured on real 2026-08 data, the step-only
    test fired 16 times per satellite on average and inflated variance up to
    45x while driving the correlation with the observed series to zero.

    Args:
        s:            the series, already reset to a clean integer index
        threshold_ns: explicit step threshold, or None to derive one
    Returns:
        Sorted list of indices where a persistent jump begins.
    """
    diffs     = s.diff()
    threshold = _jump_threshold(diffs, threshold_ns)
    centre    = _step_centre(diffs)
    excess    = diffs - centre

    candidates = excess[excess.abs() > threshold].index.tolist()
    if not candidates:
        return []

    w = LEVEL_WINDOW
    confirmed = []
    for idx in candidates:
        if idx - w < 0 or idx + w > len(s):
            continue
        before = s.iloc[idx - w:idx].astype(float)
        after  = s.iloc[idx:idx + w].astype(float)

        level_shift = float(after.median() - before.median())

        # Robust within-window noise, expressed as a sigma equivalent.
        noise = MAD_TO_SIGMA * 0.5 * (
            float((before - before.median()).abs().median())
            + float((after - after.median()).abs().median())
        )
        if noise < MIN_SPREAD:
            noise = MIN_SPREAD

        # The level must genuinely move, in the same direction as the step.
        if abs(level_shift) > LEVEL_SIGMAS * noise and \
           np.sign(level_shift) == np.sign(float(excess.iloc[idx])):
            confirmed.append((idx, float(excess.iloc[idx])))

    if not confirmed:
        return []

    # Suppress candidates that form a regular, uniform, same-signed pattern:
    # that is the routine upload sawtooth, whatever its period, and it is
    # structure for the models to learn rather than a defect to remove.
    confirmed = [c for c in confirmed
                 if c[0] not in _routine_pattern_indices(confirmed)]
    if not confirmed:
        return []

    # Separate anomalies from the routine upload sawtooth. When a satellite has
    # a population of similar, regular resets, those are its normal behaviour;
    # only a step that stands out from that population is an IOD defect.
    if len(confirmed) < MIN_ROUTINE_JUMPS:
        return _reject_systematic(confirmed, s)

    sizes  = np.array([abs(v) for _, v in confirmed])
    centre = float(np.median(sizes))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(sizes - centre)))
    if spread < MIN_SPREAD:
        # Perfectly regular resets: entirely routine, nothing anomalous.
        return []

    cutoff  = centre + ANOMALY_SIGMAS * spread
    selected = [(i, v) for (i, v) in confirmed if abs(v) > cutoff]
    return _reject_systematic(selected, s)


def _reject_systematic(selected: list, s: pd.Series) -> list:
    """
    Drop a correction set whose removals share a direction.

    Anomalies are one-offs with no preferred sign. When the signed excesses sum
    to a shift comparable to the series' own range, the steps being removed are
    the satellite's systematic behaviour rather than defects, and subtracting
    them manufactures a trend that was never in the data.

    Args:
        selected: list of (index, signed excess) chosen for correction
        s:        the series being corrected
    Returns:
        The indices to correct, or an empty list when the set is systematic.
    """
    if not selected:
        return []
    signal_range = float(s.max() - s.min())
    if signal_range <= 0:
        return [i for i, _ in selected]

    net = abs(float(sum(v for _, v in selected)))
    if net > MAX_NET_SHIFT_FRACTION * signal_range:
        return []
    return [i for i, _ in selected]


def _routine_pattern_indices(confirmed: list) -> set:
    """
    Indices belonging to a regular, same-signed, uniform-magnitude pattern.

    Such a pattern is the routine broadcast-upload sawtooth. It is recognised
    from its own regularity rather than from how many candidates there happen
    to be, so it is caught even when only two or three resets are present.

    Args:
        confirmed: list of (index, signed excess) for persistent level shifts
    Returns:
        Set of indices to suppress; empty when the pattern is not routine.
    """
    if len(confirmed) < 2:
        return set()

    idx  = np.array([i for i, _ in confirmed], dtype=float)
    vals = np.array([v for _, v in confirmed], dtype=float)

    signs = np.sign(vals)
    if not np.all(signs == signs[0]):
        return set()

    mags    = np.abs(vals)
    mag_med = float(np.median(mags))
    if mag_med <= 0:
        return set()
    if float(np.median(np.abs(mags - mag_med))) / mag_med > ROUTINE_MAG_TOLERANCE:
        return set()

    if len(idx) == 2:
        # Two same-signed resets of the same size: too little to call a rhythm,
        # but equally too little to justify removing them as anomalies.
        return set(int(i) for i in idx)

    gaps    = np.diff(idx)
    gap_med = float(np.median(gaps))
    if gap_med <= 0:
        return set()
    if float(np.median(np.abs(gaps - gap_med))) / gap_med > ROUTINE_GAP_TOLERANCE:
        return set()

    return set(int(i) for i in idx)


def count_jumps(series: pd.Series, threshold_ns: float | None = None) -> int:
    """
    Count IOD discontinuities without modifying the series.

    Args:
        series:       satellite error time series
        threshold_ns: explicit threshold, or None to derive one
    Returns:
        Number of steps classified as IOD jumps.
    """
    s = pd.Series(series).astype(float).reset_index(drop=True)
    return len(_persistent_jump_indices(s, threshold_ns))


def correct_iod_jumps(
    series: pd.Series,
    threshold_ns: float | None = None,
) -> pd.Series:
    """
    Detect and remove IOD upload discontinuities from a satellite error series.

    At each step whose departure from the satellite's typical step is anomalous,
    the ANOMALOUS EXCESS is subtracted from all subsequent values, making the
    series continuous. Normal drift -- however fast -- is left untouched,
    including across a corrected jump: only the discontinuity is removed, not
    the continuous change the satellite would have undergone anyway.

    Note that the result lives in a shifted coordinate frame whenever any jump
    is corrected. Callers reconstructing forecasts to original units must
    anchor on the observed series, not on this one; see
    preprocess_satellite()'s 'observed' output.

    Args:
        series:       satellite error time series (clock in ns, ephemeris in m)
        threshold_ns: explicit jump threshold. None (the default) derives one
                      from the robust spread of this satellite's own steps.
    Returns:
        Corrected series with the same index as the input.
    """
    s = pd.Series(series).copy().astype(float).reset_index(drop=True)
    centre = _step_centre(s.diff())

    for idx in _persistent_jump_indices(s, threshold_ns):
        # Remove only the part of the step that is not ordinary drift.
        excess = float(s.iloc[idx] - s.iloc[idx - 1]) - centre
        s.iloc[idx:] -= excess

    s.index = pd.Series(series).index
    return s
