"""
The data-ingest contract — the single loader every lane consumes.

Read this before writing any CSV-reading code. The point of this module is that
nobody else writes a loader. Five people writing five loaders is how one of them
silently merges the two satellites stacked inside each MEO file, and nobody
notices because the row count still looks plausible.

What the files actually are
---------------------------
The organisers ship GEO in one file and MEO in two files, at a non-uniform
sampling rate (Note.pdf). What is not stated, and what a naive loader gets
wrong, is that **each MEO file contains two satellite series stacked
vertically**. Both blocks cover the same time span, so the file's timestamp
column runs forward, jumps *backwards* once, then runs forward again:

    DATA_MEO_Train.csv    46 rows fwd  | backward jump | 44 rows fwd   -> 2 series
    DATA_MEO_Train2.csv  143 rows fwd  | backward jump | 101 rows fwd  -> 2 series
    DATA_GEO_Train.csv   142 rows fwd, no backward jump                -> 1 series

The single backward jump is the block boundary. `split_stacked_blocks` cuts
there. GEO has no jump and stays one series. This yields five series in total
(GEO + two per MEO file), which is what the whole team means by "the five
series".

The bug this replaces
---------------------
`run_pipeline.py` labels every row of a MEO file with a single SatelliteID
("MEO"), so the two stacked blocks are treated as one series with a six-day
backward step in the middle of it. `test_ingest_contract.py` pins the correct
behaviour so that bug cannot come back through this path.

Units
-----
All four targets are already in metres in the shipped files (the clock error
was converted by the organisers via e_clock,m = e_clock * c). This loader does
not rescale anything; it only parses, splits and labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The four target columns, in canonical order, as they appear after
# normalisation. This order matches evaluation.scoring.PARAMETERS and the
# (n, 4) array the prediction contract returns.
TARGET_COLUMNS: tuple[str, ...] = (
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
)

# Short parameter names aligned 1:1 with TARGET_COLUMNS.
PARAMETERS: tuple[str, ...] = ("x_error", "y_error", "z_error", "satclockerror")

TIME_COLUMN = "utc_time"


def _canonical(name: str) -> str:
    """Collapse header whitespace/case so 'y_error  (m)' == 'y_error (m)'.

    The shipped CSVs are inconsistent: DATA_MEO_Train.csv has a double space in
    'y_error  (m)'. Matching on a normalised key keeps the loader from silently
    dropping a target column because of a stray space.
    """
    return " ".join(name.strip().lower().split())


# Normalised header -> canonical target column.
_TARGET_LOOKUP = {_canonical(c): c for c in TARGET_COLUMNS}
_TIME_KEYS = {_canonical(k) for k in (TIME_COLUMN, "time", "timestamp")}


@dataclass(frozen=True)
class Series:
    """One satellite's error series — the unit every lane works on.

    Attributes:
        satellite_id: unique id, e.g. 'MEO_Train-b0'. Stable across a run.
        dataset:      source dataset tag, e.g. 'GEO_Train', 'MEO_Train2'.
        orbit:        'GEO' or 'MEO'.
        block:        0-based index of the stacked block within the file.
        frame:        DataFrame with a 'Timestamp' column (datetime64, strictly
                      increasing) followed by the four TARGET_COLUMNS in metres.
    """
    satellite_id: str
    dataset: str
    orbit: str
    block: int
    frame: pd.DataFrame

    @property
    def times(self) -> pd.Series:
        """The Timestamp column."""
        return self.frame["Timestamp"]

    @property
    def n(self) -> int:
        """Number of rows in the series."""
        return len(self.frame)

    def values(self) -> np.ndarray:
        """Return the four targets as an (n, 4) float array, in TARGET_COLUMNS order."""
        return self.frame[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)


def _read_and_normalise(path: str | Path) -> pd.DataFrame:
    """Read a CSV and return a frame with a parsed Timestamp and the 4 targets.

    Args:
        path: CSV path.
    Returns:
        DataFrame with columns ['Timestamp', *TARGET_COLUMNS], row order
        preserved from the file (NOT yet split or sorted).
    Raises:
        ValueError: if the time column or any target column is missing.
    """
    raw = pd.read_csv(path)
    rename: dict[str, str] = {}
    time_src: str | None = None
    for col in raw.columns:
        key = _canonical(col)
        if key in _TIME_KEYS:
            time_src = col
        elif key in _TARGET_LOOKUP:
            rename[col] = _TARGET_LOOKUP[key]

    if time_src is None:
        raise ValueError(f"{path}: no time column found among {list(raw.columns)}")
    df = raw.rename(columns={time_src: "Timestamp", **rename})

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing target columns {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df[["Timestamp", *TARGET_COLUMNS]].reset_index(drop=True)


def split_stacked_blocks(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a file's rows into blocks at each backward time step.

    A block boundary is any row whose Timestamp is not strictly greater than the
    previous row's — i.e. the point where a second stacked satellite restarts
    the clock. GEO files have no such step and return a single block.

    Args:
        frame: normalised frame with a 'Timestamp' column, in file row order.
    Returns:
        List of frames, one per stacked series, each strictly time-increasing.
    """
    t = frame["Timestamp"].to_numpy()
    # Boundary where the clock does not advance (backward or equal step).
    boundaries = np.where(np.diff(t) <= np.timedelta64(0))[0] + 1
    starts = [0, *boundaries.tolist()]
    stops = [*boundaries.tolist(), len(frame)]
    return [frame.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, stops)]


def _orbit_and_tag(path: Path) -> tuple[str, str]:
    """Derive orbit type and a dataset tag from the file name.

    The organisers split by orbit at the file level (Note.pdf), so the file name
    is the authoritative source of orbit type here.

    Args:
        path: CSV path.
    Returns:
        (orbit, dataset_tag), e.g. ('MEO', 'MEO_Train2').
    """
    stem = path.stem  # e.g. DATA_MEO_Train2
    tag = stem[len("DATA_"):] if stem.upper().startswith("DATA_") else stem
    orbit = "GEO" if "GEO" in stem.upper() else "MEO"
    return orbit, tag


def load_series(path: str | Path) -> list[Series]:
    """Load one CSV into its constituent satellite series.

    A GEO file yields one Series; each MEO file yields two (the stacked blocks).

    Args:
        path: CSV path.
    Returns:
        List of Series, each with a strictly time-increasing frame.
    Raises:
        ValueError: on a malformed header (see _read_and_normalise).
    """
    path = Path(path)
    df = _read_and_normalise(path)
    orbit, tag = _orbit_and_tag(path)
    blocks = split_stacked_blocks(df)
    return [
        Series(
            satellite_id=f"{tag}-b{i}",
            dataset=tag,
            orbit=orbit,
            block=i,
            frame=block,
        )
        for i, block in enumerate(blocks)
    ]


def load_dataset(paths: list[str | Path]) -> list[Series]:
    """Load several CSVs into a flat list of series.

    Args:
        paths: CSV paths, typically the three training files.
    Returns:
        The concatenation of load_series over each path, order preserved.
    """
    out: list[Series] = []
    for p in paths:
        out.extend(load_series(p))
    return out
