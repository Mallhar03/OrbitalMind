"""
Data-driven train / calibration / forecast windows.

The pipeline used to hardcode the indices 480, 576 and 672. That only ever
matched one file shape: the 768-row (8-day) output of our own synthetic
generator. Two failures followed from it.

  * On the organisers' format -- "~672 rows per satellite", i.e. 7 days at
    15-minute spacing -- the window [576:672] falls *inside* the file. The
    pipeline re-predicted day 7 and labelled it day 8.
  * On any longer record every row past index 672 was discarded. The real
    IGS data (1344 rows/satellite) lost 671 of them.

compute_splits derives the windows from the actual series length, so the same
code is correct for a 7-day file, our 8-day synthetic file, and a 14-day real
record. All indices are half-open [start, stop) in *differenced* index space,
which is one shorter than the raw satellite record.

Two plans are produced:

  backtest    Scores the model honestly. The final `horizon` steps are held
              out as ground truth and never seen during training or
              calibration.

  submission  Produces the deliverable. The forecast window starts at the end
              of the record, because day 8 does not exist in a 7-day file.
              Calibration still sits outside the training window, so residual
              statistics stay out-of-sample.
"""
from dataclasses import dataclass

SEQ_LEN = 96   # 15-minute steps in 24 hours
HORIZON = 96   # steps to forecast (day 8, at 15-minute intervals)

MIN_LENGTH_MSG = "record too short for the train/calibration/hold-out split"

Window = tuple[int, int]


@dataclass(frozen=True)
class Plan:
    """One set of windows: what to train on, calibrate on, feed in, and predict.

    Attributes:
        train:     window the base models fit on
        cal:       full calibration window, disjoint from train
        cal_input: seq_len steps feeding the calibration-window forecast
        cal_meta:  first half of cal — fits the LightGBM meta-learner
        cal_flow:  second half of cal — fits the residual flow, and is
                   out-of-sample for the meta-learner as well as the base
                   models, so the learned spread is not optimistic
        input:     seq_len steps fed to the models to produce the forecast
        target:    window being predicted; for the submission plan this lies
                   past the end of the record and has no ground truth
    """
    train:     Window
    cal:       Window
    cal_input: Window
    cal_meta:  Window
    cal_flow:  Window
    input:     Window
    target:    Window


@dataclass(frozen=True)
class Splits:
    """The backtest and submission plans for one satellite series."""
    n:        int
    seq_len:  int
    horizon:  int
    backtest: Plan
    submission: Plan


def compute_splits(n: int, seq_len: int = SEQ_LEN, horizon: int = HORIZON) -> Splits:
    """
    Derive train/calibration/forecast windows from the actual series length.

    Layout, where n is the differenced series length and h the horizon:

        backtest    train [0, n-2h)   cal [n-2h, n-h)   target [n-h, n)
        submission  train [0, n-h)    cal [n-h, n)      target [n, n+h)

    The backtest target is real held-out data. The submission target extends
    past the record, which is what "predict day 8 from 7 days of history"
    actually requires.

    Args:
        n:       length of the differenced series (raw row count minus one)
        seq_len: number of history steps fed to the models
        horizon: number of steps to forecast
    Returns:
        Splits carrying both plans.
    Raises:
        ValueError: if n is too short to leave every model a trainable window.
    """
    minimum = seq_len + 3 * horizon
    if n < minimum:
        raise ValueError(
            f"{MIN_LENGTH_MSG}: got {n} differenced steps, need at least "
            f"{minimum} (seq_len {seq_len} + 3 x horizon {horizon}). "
            f"That is {minimum + 1} rows per satellite."
        )

    def plan(train_stop: int, cal_start: int, cal_stop: int,
             input_start: int, target_start: int) -> Plan:
        """Assemble one plan, splitting its calibration window in half."""
        mid = cal_start + (cal_stop - cal_start) // 2
        return Plan(
            train     = (0,                     train_stop),
            cal       = (cal_start,             cal_stop),
            cal_input = (cal_start - seq_len,   cal_start),
            cal_meta  = (cal_start,             mid),
            cal_flow  = (mid,                   cal_stop),
            input     = (input_start,           input_start + seq_len),
            target    = (target_start,          target_start + horizon),
        )

    backtest = plan(
        train_stop   = n - 2 * horizon,
        cal_start    = n - 2 * horizon,
        cal_stop     = n - horizon,
        input_start  = n - horizon - seq_len,
        target_start = n - horizon,
    )
    submission = plan(
        train_stop   = n - horizon,
        cal_start    = n - horizon,
        cal_stop     = n,
        input_start  = n - seq_len,
        target_start = n,
    )
    return Splits(
        n=n, seq_len=seq_len, horizon=horizon,
        backtest=backtest, submission=submission,
    )
