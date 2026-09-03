"""Shared training utilities and RMSE evaluation for all GNSS models."""
import numpy as np


def compute_rmse_horizons(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE at six prediction horizons, in two different senses.

    Horizon-to-step mapping (at 15-minute intervals):
        15min → step 1,   30min → step 2,   1hr  → step 4,
        2hr   → step 8,   12hr  → step 48,  24hr → step 96

    Two families of numbers are returned for those same six horizons.

    CUMULATIVE keys ('15min' … '24hr') average the squared error over every
    step from 1 up to and including the horizon. '24hr' is therefore the
    error over the whole forecast day, not the error one day out.

    TERMINAL keys ('at_15min' … 'at_24hr') use one step alone: the step at
    the horizon itself. 'at_24hr' is the error of the very last predicted
    point, 24 hours after the forecast starts.

    A cumulative figure ALWAYS UNDERSTATES the terminal error whenever error
    grows with lead time, which it does for every model here. Its 96-step
    average is pulled down by the 95 shorter, easier steps in front of it, so
    '24hr' is not, and must never be quoted as, "the error at 24 hours" —
    that number is 'at_24hr'. The two are equal only at '15min'/'at_15min',
    where the horizon is a single step anyway.

    Short inputs are handled differently in each family, deliberately:

      - A cumulative horizon longer than the data averages over whatever is
        available (existing behaviour, unchanged). Given 4 steps, '24hr' is
        the RMSE over those 4 steps.
      - A terminal horizon longer than the data is NaN. There is no step 96
        in a 4-step array, and clamping to the last available step would
        return a small, plausible-looking number that a reader would take for
        a genuine 24-hour error. NaN cannot be misread that way, and it
        propagates rather than quietly flattering an average.

    Inputs are flattened first (existing behaviour), so a multi-dimensional
    array is treated as one flat sequence and "step" counts elements of that
    flattened sequence.

    Args:
        y_true: array of ground-truth values
        y_pred: array of predicted values (same or greater length as y_true)
    Returns:
        Dict of float RMSEs, in this insertion order: the six cumulative keys
        '15min', '30min', '1hr', '2hr', '12hr', '24hr', followed by the six
        terminal keys 'at_15min', 'at_30min', 'at_1hr', 'at_2hr', 'at_12hr',
        'at_24hr'. A terminal key is NaN when the data is shorter than its
        horizon. Callers that print a report rely on this order.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))

    horizon_steps = {"15min": 1, "30min": 2, "1hr": 4,
                     "2hr": 8, "12hr": 48, "24hr": 96}
    result = {}
    for key, step in horizon_steps.items():
        k = min(step, n)
        result[key] = float(np.sqrt(np.mean((y_true[:k] - y_pred[:k]) ** 2)))
    for key, step in horizon_steps.items():
        if step > n:
            result[f"at_{key}"] = float("nan")
            continue
        # Slice rather than index a scalar: the same sqrt-of-mean-square form
        # as above, so a single step stays correct for any input shape.
        diff = y_true[step - 1:step] - y_pred[step - 1:step]
        result[f"at_{key}"] = float(np.sqrt(np.mean(diff ** 2)))
    return result
