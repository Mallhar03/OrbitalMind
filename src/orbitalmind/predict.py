"""
The single inference entrypoint — the real deliverable.

Given the organisers' 7-day training file(s) and a set of query timestamps,
fit fresh models, pick the best one per series WITHOUT looking at the answer,
and forecast the four errors (x, y, z, clock) at those timestamps. If the
timestamp file also carries truth columns, the residuals are scored with the
admissible Shapiro-Francia scorer and a priority-1/2 report is written.

Why this file exists
--------------------
The final evaluation (Note.pdf 1c-1e) is exactly this shape: the organisers
hand each team a fresh 7-day record and a list of arbitrary 8th-day timestamps,
and score the residuals themselves. Both possible forms of what they bring — a
bare timestamp list, or a new test file — reduce to "fit on these 7 days,
predict at these timestamps". So this one path covers both, and it never
depends on the current day-8 file the way scripts/generate_submission.py did.

Leak safety
-----------
Model *selection* is the only place the answer could leak in, and it does not
here. Each series' 7-day training record is split in time: the models are
fitted on the earlier part and ranked on a held-out tail of the SAME training
record — never on the query/test truth. The winner is then refit on the full
7 days before it forecasts. Nothing that steers the choice of model has seen a
value the organisers will score.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from orbitalmind.ingest import (
    Series,
    load_dataset,
    TARGET_COLUMNS,
    PARAMETERS,
    _read_and_normalise,
    split_stacked_blocks,
    _orbit_and_tag,
)
from orbitalmind.interfaces import predict_frame
from orbitalmind.evaluation.scoring import score_residuals, ALPHA
from orbitalmind.models.harmonic import HarmonicPredictor
from orbitalmind.models.gp_predictor import GaussianProcessPredictor
from orbitalmind.models.deep_predictor import DeepResidualPredictor

# Fraction of each training record's tail held out, in time, for leak-free
# model selection. 0.25 of 7 days ~= the final ~1.75 days.
VALIDATION_TAIL_FRACTION = 0.25

# The candidate model factories. Each returns a fresh, unfitted predictor.
# Kept deterministic (seeded) so a run is reproducible.
CANDIDATES: dict[str, callable] = {
    "Harmonic": lambda: HarmonicPredictor(),
    "GP": lambda: GaussianProcessPredictor(n_restarts=2, random_state=42),
    "Deep": lambda: DeepResidualPredictor(epochs=40, lr=0.003, weight_decay=1e-3),
}


@dataclass
class SeriesResult:
    """What the entrypoint produces for one satellite series."""

    satellite_id: str
    orbit: str
    chosen_model: str
    validation_W: float  # leak-free selection score
    predictions: pd.DataFrame  # Timestamp + 4 target columns (raw model output)
    score: dict | None  # priority-1/2 report if truth was given
    calibration: object = None  # shaping.Calibration (sigma, source)


def _slice_series(series: Series, lo: int, hi: int) -> Series:
    """Return a Series backed by rows [lo, hi) of the given series."""
    frame = series.frame.iloc[lo:hi].reset_index(drop=True)
    return Series(
        satellite_id=series.satellite_id,
        dataset=series.dataset,
        orbit=series.orbit,
        block=series.block,
        frame=frame,
    )


def _fit(name: str, train: Series):
    """Fit one named candidate on a training Series and return it."""
    return CANDIDATES[name]().fit(train)


def _residuals_on(model, truth: Series) -> dict[str, np.ndarray]:
    """Per-parameter residuals (prediction - truth) at the truth's timestamps."""
    preds = np.asarray(model.predict(list(truth.times)), dtype=np.float64)
    actual = truth.values()
    return {PARAMETERS[j]: preds[:, j] - actual[:, j] for j in range(len(PARAMETERS))}


def select_model(train: Series) -> tuple[str, float]:
    """Choose the best candidate for a series without touching the query truth.

    The training record is split in time: fit on the earlier part, rank on the
    held-out tail of the SAME training record. This is the anti-leak core — the
    ranking never sees a value the organisers will score.

    Args:
        train: the full 7-day training Series.
    Returns:
        (chosen_model_name, validation_W). Falls back to 'Harmonic' if the
        series is too short to split.
    """
    n = train.n
    cut = int(round(n * (1.0 - VALIDATION_TAIL_FRACTION)))
    # Need enough rows on both sides to fit and to score (SF needs >= 3).
    if cut < 5 or (n - cut) < 4:
        return "Harmonic", float("nan")

    fit_part = _slice_series(train, 0, cut)
    val_part = _slice_series(train, cut, n)

    best_name, best_W = None, -np.inf
    for name in CANDIDATES:
        try:
            model = _fit(name, fit_part)
            resids = _residuals_on(model, val_part)
            W = score_residuals(resids).W
        except Exception:
            continue
        if W > best_W:
            best_name, best_W = name, W

    if best_name is None:  # every candidate failed
        return "Harmonic", float("nan")
    return best_name, float(best_W)


def forecast_series(train: Series, t_query: Sequence) -> tuple[SeriesResult, object]:
    """Select, refit on the full 7 days, and forecast at the query timestamps.

    Args:
        train: the 7-day training Series.
        t_query: query timestamps (arbitrary, need not be uniform or sorted).
    Returns:
        (SeriesResult without its score yet, the fitted model).
    """
    from orbitalmind.shaping import calibrate  # local import avoids a cycle

    name, val_W = select_model(train)
    model = _fit(name, train)  # refit winner on the FULL record
    pred_df = predict_frame(model, t_query)  # the submitted point forecast, untouched

    # Calibrate on training only (leak-free). This supplies the predictive
    # interval's sigma; it does NOT modify the point forecast above (see
    # shaping.py), so it cannot affect W or the submitted predictions.
    cal = calibrate(name, train)

    result = SeriesResult(
        satellite_id=train.satellite_id,
        orbit=train.orbit,
        chosen_model=name,
        validation_W=val_W,
        predictions=pred_df,
        score=None,
        calibration=cal,
    )
    return result, model


def _load_query(
    timestamps_path: str,
) -> tuple[list, dict[str, np.ndarray] | None, list]:
    """Load the query file: timestamps, optional per-series key, optional truth.

    The file must have a time column. If it also carries the four target
    columns, they are returned as truth so residuals can be scored. If it has a
    'satellite_id' column, predictions are matched to that series; otherwise all
    timestamps are applied to every series.

    Returns:
        (timestamps, truth_or_None, satellite_ids_or_empty).
    """
    raw = pd.read_csv(timestamps_path)
    lower = {c.lower().strip(): c for c in raw.columns}
    # time column
    time_col = next(
        (lower[k] for k in ("utc_time", "time", "timestamp") if k in lower), None
    )
    if time_col is None:
        raise ValueError(f"{timestamps_path}: no time column (utc_time/time/timestamp)")
    times = pd.to_datetime(raw[time_col], format="mixed")

    sat_col = lower.get("satellite_id")
    sat_ids = raw[sat_col].tolist() if sat_col else []

    # optional truth
    from orbitalmind.ingest import _canonical  # local import to avoid cycle noise

    tgt_lookup = {_canonical(c): c for c in TARGET_COLUMNS}
    have = {
        tgt_lookup[_canonical(c)]: c for c in raw.columns if _canonical(c) in tgt_lookup
    }
    truth = None
    if len(have) == len(TARGET_COLUMNS):
        truth = {
            PARAMETERS[j]: raw[have[TARGET_COLUMNS[j]]].to_numpy(dtype=float)
            for j in range(len(PARAMETERS))
        }
    return list(times), truth, sat_ids


def run(
    train_files: list[str], timestamps_path: str, qq_dir: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Fit on the training file(s) and forecast at the query timestamps.

    Args:
        train_files: one or more 7-day training CSVs (GEO and/or MEO).
        timestamps_path: CSV with a time column, optionally a satellite_id
            column, optionally the four truth columns for scoring.
        qq_dir: if given and the query carried truth, write a per-series Q-Q
            plot (priority-3 deliverable) into this directory.
    Returns:
        (submission DataFrame, report dict). The report carries per-series
        chosen models and, when truth was supplied, the priority-1/2 scores.
    """
    train_series = load_dataset(train_files)
    t_query, truth, sat_ids = _load_query(timestamps_path)
    if qq_dir:
        os.makedirs(qq_dir, exist_ok=True)

    frames, per_series, pooled = [], [], {p: [] for p in PARAMETERS}
    for train in train_series:
        result, model = forecast_series(train, t_query)
        df = result.predictions.copy()
        df.insert(1, "satellite_id", train.satellite_id)
        df.insert(2, "orbit", train.orbit)
        # Predictive interval (point +/- 1.96 sigma) as extra columns.
        point = result.predictions[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
        low, high = result.calibration.interval(point)
        for j, col in enumerate(TARGET_COLUMNS):
            base = col[: col.rfind(" (")] if " (" in col else col  # e.g. 'x_error'
            df[f"{base}_lower (m)"] = low[:, j]
            df[f"{base}_upper (m)"] = high[:, j]
        frames.append(df)

        cal = result.calibration
        entry = {
            "satellite_id": train.satellite_id,
            "orbit": train.orbit,
            "chosen_model": result.chosen_model,
            "validation_W": result.validation_W,
            "calibration": {
                "source": cal.source,
                "sigma": {
                    PARAMETERS[j]: float(cal.sigma[j]) for j in range(len(PARAMETERS))
                },
            },
        }
        # Score only if the query file carried matching truth (same length).
        if truth is not None and len(t_query) == len(next(iter(truth.values()))):
            actual = np.column_stack([truth[p] for p in PARAMETERS])
            preds = result.predictions[list(TARGET_COLUMNS)].to_numpy(float)
            resid = {
                PARAMETERS[j]: preds[:, j] - actual[:, j]
                for j in range(len(PARAMETERS))
            }
            sc = score_residuals(resid)
            entry["score"] = sc.as_dict()
            entry["gaussian"] = "PASS" if sc.H == 0 else "FAIL"
            for p in PARAMETERS:
                pooled[p].extend(resid[p])
            if qq_dir:
                from orbitalmind.evaluation.gaussian_check import save_qq_plot

                allr = np.concatenate([resid[p] for p in PARAMETERS])
                save_qq_plot(allr, os.path.join(qq_dir, f"qq_{train.satellite_id}.png"))
        per_series.append(entry)

    submission = pd.concat(frames, ignore_index=True)
    report = {"alpha": ALPHA, "n_series": len(train_series), "series": per_series}
    if any(len(v) for v in pooled.values()):
        report["overall"] = score_residuals(pooled).as_dict()
        report["gaussian_pass_count"] = sum(
            1 for s in per_series if s.get("gaussian") == "PASS"
        )
        if qq_dir:
            from orbitalmind.evaluation.gaussian_check import save_qq_plot

            save_qq_plot(
                np.concatenate([np.asarray(pooled[p]) for p in PARAMETERS]),
                os.path.join(qq_dir, "qq_pooled.png"),
            )
    return submission, report


def _format_report(report: dict) -> str:
    """Render the report dict as the human-readable priority-1/2 summary."""
    lines = [
        f"OrbitalMind forecast report  (alpha={report['alpha']})",
        f"{report['n_series']} series",
        "",
        f"{'series':<20}{'orbit':<6}{'model':<10}{'val_W':>8}{'W':>8}{'p':>8}{'H':>4}{'result':>8}",
    ]
    for s in report["series"]:
        W = p = H = res = "-"
        if "score" in s:
            W = f"{s['score']['W']:.4f}"
            p = f"{s['score']['p_value']:.4f}"
            H = str(s["score"]["H"])
            res = s.get("gaussian", "-")
        vW = (
            f"{s['validation_W']:.4f}"
            if s["validation_W"] == s["validation_W"]
            else "nan"
        )
        lines.append(
            f"{s['satellite_id']:<20}{s['orbit']:<6}{s['chosen_model']:<10}{vW:>8}{W:>8}{p:>8}{H:>4}{res:>8}"
        )
    if "overall" in report:
        o = report["overall"]
        pc = report.get("gaussian_pass_count")
        pass_line = (
            f"  ({pc}/{report['n_series']} series pass normality)"
            if pc is not None
            else ""
        )
        lines += [
            "",
            f"OVERALL  W={o['W']:.4f}  p={o['p_value']:.4f}  H={o['H']}  (H=0 means residuals pass as Gaussian at alpha){pass_line}",
        ]
        lines.append(
            f"{'param':<16}{'W':>8}{'p':>8}{'H':>4}{'mean':>12}{'std':>12}{'ci_low':>12}{'ci_high':>12}"
        )
        for name, ps in o["per_parameter"].items():
            lines.append(
                f"{name:<16}{ps['W']:>8.4f}{ps['p_value']:>8.4f}{ps['H']:>4}"
                f"{ps['mean']:>12.4f}{ps['std']:>12.4f}{ps['ci_low']:>12.4f}{ps['ci_high']:>12.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="OrbitalMind inference entrypoint")
    parser.add_argument(
        "--train", nargs="+", required=True, help="7-day training CSV(s)"
    )
    parser.add_argument(
        "--timestamps",
        required=True,
        help="CSV with a time column (optionally satellite_id and truth columns)",
    )
    parser.add_argument(
        "--output", default="outputs/submission.csv", help="where to write predictions"
    )
    parser.add_argument(
        "--report",
        default="outputs/forecast_report.txt",
        help="where to write the report",
    )
    parser.add_argument(
        "--qq-dir",
        default="outputs/qq",
        help="directory for Q-Q plots (when truth is supplied)",
    )
    args = parser.parse_args()

    submission, report = run(args.train, args.timestamps, qq_dir=args.qq_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    submission.to_csv(args.output, index=False)
    with open(args.report, "w") as fh:
        fh.write(_format_report(report))
    with open(os.path.splitext(args.report)[0] + ".json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(_format_report(report))
    print(f"\nPredictions -> {args.output}")
    print(f"Report      -> {args.report}")


if __name__ == "__main__":
    main()
