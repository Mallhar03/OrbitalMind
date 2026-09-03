#!/usr/bin/env python3
"""
Day-6 ablation — find out which parts of the ensemble earn their place.

The pipeline fuses four base models, optionally adds engineered features, and
throws away roughly half the decomposed signal. None of those three choices has
ever been measured. This script measures all of them.

UNITS — THESE NUMBERS RANK CONFIGURATIONS, THEY DO NOT MEASURE ACCURACY

Scoring happens in DIFFERENCED space, the models' own working space, because that
is where the fusion operates and where a like-for-like comparison between
configurations is cleanest. run_pipeline.py scores differently — it reconstructs to
original units from an anchor before measuring.

So these figures are valid for saying "configuration A beats configuration B" and
invalid as an absolute accuracy claim. They must never be compared to the deck's
0.65 ns or 7.5 ns targets, which are stated on the reconstructed error.

WHAT IT SCORES AGAINST, AND WHY THAT MATTERS

Every number here comes from the BACKTEST window — the last 24 hours of the
7-day input, held out from training but part of the data we were given. The
2026-08-28 holdout file is never opened. That distinction is the whole point:
choosing a configuration by comparing scores on the holdout would be tuning
against the answer we are meant to predict, and the result would be worthless
however good it looked.

WHAT IT MEASURES

1. Leave-one-out over the four base models. Each is dropped from the fusion in
   turn, the meta-learner is refitted on the remainder, and the forecast is
   rescored. A model that earns its place makes the error worse when removed.
   Base models are trained ONCE per satellite and reused across these
   configurations, because dropping a model from the fusion does not change how
   the others were fitted.

   THIS SCRIPT NEVER CHANGES THE PIPELINE. It reports; it does not prune. All four
   models are retained unconditionally by Decision 020 — they are the core of the
   solution and the Neural ODE is deck claim C-08. A leave-one-out row showing a
   model contributes little is PREPARATION for the question "what does this model
   add?", and often a symptom of that model being undertrained or badly fed. It is
   never an argument for removing it.

2. Engineered features on and off — the comparison plan/day-04 asks for and
   explicitly defers to day 5.

3. The discarded noise IMF, in and out. This is the open question S3 handed
   forward: the discarded component is 33-77% of the variance and is not clean
   noise, but most of its energy is broadband. Including it requires retraining
   the base models on a different signal, so it is a separate pass.

Usage:
    python scripts/ablation.py --satellites 6
    python scripts/ablation.py --satellites 6 --skip-imf   # faster, no pass 3

Output:
    outputs/ablation_report.txt
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.splits import compute_splits, SEQ_LEN, HORIZON  # noqa: E402
from orbitalmind.preprocessing.pipeline import preprocess_satellite  # noqa: E402
from orbitalmind.models.base_trainer import compute_rmse_horizons  # noqa: E402
from orbitalmind.ensemble.lightgbm_meta import (  # noqa: E402
    train_meta_learner, predict_meta_learner,
)
from orbitalmind import run_pipeline as rp  # noqa: E402

BASE_MODELS = ("lstm", "tcn_lstm", "tft", "neural_ode")
INPUT_CSV = "data/raw/gnss_real.csv"
REPORT = "outputs/ablation_report.txt"


def _fuse_and_score(cal_outputs, tgt_outputs, cal_truth, truth, plan, combined,
                    keep, use_features):
    """
    Refit the meta-learner on a subset of base models and score the forecast.

    Args:
        cal_outputs: base model forecasts over the calibration window
        tgt_outputs: base model forecasts over the target window
        cal_truth:   truth over the calibration window
        truth:       truth over the target window, in differenced space
        plan:        the backtest Plan
        combined:    the full differenced signal
        keep:        base model names to fuse
        use_features: whether to add the engineered features
    Returns:
        RMSE at the 1-hour horizon, or None if the fit failed.
    """
    m0, m1 = plan.cal_meta[0] - plan.cal[0], plan.cal_meta[1] - plan.cal[0]

    meta_in = {k: cal_outputs[k][m0:m1] for k in keep}
    tgt_in = {k: tgt_outputs[k] for k in keep}
    if use_features:
        meta_in.update(rp._meta_features(combined, plan.cal[0] + m0,
                                         plan.cal[0] + m1, plan.cal_input[1]))
        tgt_in.update(rp._meta_features(combined, plan.target[0], plan.target[1],
                                        plan.input[1]))
    try:
        meta = train_meta_learner(meta_in, cal_truth[m0:m1], "ABL", "ablation")
        pred = predict_meta_learner(meta, tgt_in)
    except Exception:
        return None
    return compute_rmse_horizons(truth, pred)


def _one_satellite(df, sat_id, error_col, include_noise, skip_imf):
    """
    Run every fusion configuration for one satellite.

    Args:
        df:            full input dataframe
        sat_id:        satellite to evaluate
        error_col:     error column to model
        include_noise: add the discarded noise IMF to the modelling signal
        skip_imf:      unused here; kept so the caller reads clearly
    Returns:
        Dict mapping configuration name -> RMSE-by-horizon dict.
    """
    pre = preprocess_satellite(df[df.SatelliteID == sat_id].copy(), sat_id, error_col)
    combined = pre["trend"] + pre["periodic"]
    if include_noise:
        combined = combined + pre["noise"]
    cleaned = pre["observed"]
    splits = compute_splits(len(combined))
    plan = splits.backtest

    # Truth for the backtest target, in the differenced space the models work in.
    truth_diff = rp._slice(combined, plan.target)
    cal_truth = rp._slice(combined, plan.cal)

    np.random.seed(42)
    models = rp._train_base_models(rp._slice(combined, plan.train), "ABL",
                                   error_col, model_tag=f"abl_{sat_id}")
    cal_outputs = rp._base_forecasts(models, rp._slice(combined, plan.cal_input),
                                     len(cal_truth))
    tgt_outputs = rp._base_forecasts(models, rp._slice(combined, plan.input),
                                     plan.target[1] - plan.target[0])

    results = {}
    tag = "with-noise" if include_noise else "baseline"

    for use_feat in (False, True):
        suffix = "+features" if use_feat else ""
        r = _fuse_and_score(cal_outputs, tgt_outputs, cal_truth, truth_diff,
                            plan, combined, BASE_MODELS, use_feat)
        results[f"{tag}{suffix}"] = r

    if not include_noise:
        # Leave-one-out only needs doing once; it is a property of the fusion.
        for drop in BASE_MODELS:
            keep = tuple(m for m in BASE_MODELS if m != drop)
            r = _fuse_and_score(cal_outputs, tgt_outputs, cal_truth, truth_diff,
                                plan, combined, keep, False)
            results[f"drop-{drop}"] = r
    return results


def main():
    ap = argparse.ArgumentParser(description="Day-6 ablation on validation data.")
    ap.add_argument("--satellites", type=int, default=6,
                    help="how many satellites to evaluate")
    ap.add_argument("--data", default=INPUT_CSV)
    ap.add_argument("--skip-imf", action="store_true",
                    help="skip the noise-IMF pass, which doubles training time")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")

    # Sample across both orbit types so the GEO branch is represented — there are
    # only 7 GEO satellites against 88 MEO, so a random sample would likely miss them.
    geo = sorted(df[df.OrbitType == "GEO"].SatelliteID.unique())
    meo = sorted(df[df.OrbitType == "MEO"].SatelliteID.unique())
    n_geo = max(1, args.satellites // 2)
    chosen = geo[:n_geo] + meo[:args.satellites - n_geo]

    print(f"Ablation on {len(chosen)} satellites: {chosen}")
    print("Scoring against the BACKTEST window. The 2026-08-28 holdout is not opened.\n")

    started = time.time()
    collected = {}
    for i, sat in enumerate(chosen, 1):
        for include_noise in ((False,) if args.skip_imf else (False, True)):
            label = "with noise IMF" if include_noise else "baseline"
            print(f"  [{i}/{len(chosen)}] {sat} ({label}) ...", flush=True)
            try:
                res = _one_satellite(df, sat, "ClockError_ns", include_noise,
                                     args.skip_imf)
            except Exception as exc:
                print(f"      failed: {exc}")
                continue
            for cfg, rmse in res.items():
                if rmse is not None:
                    collected.setdefault(cfg, []).append(rmse)

    _write_report(collected, chosen, time.time() - started, args.skip_imf)


def _write_report(collected, satellites, elapsed, skipped_imf):
    """Write the ablation results, ranked, with an explicit reading of them."""
    os.makedirs("outputs", exist_ok=True)

    horizons = ["15min", "30min", "1hr", "2hr", "24hr"]
    rows = []
    for cfg, entries in collected.items():
        means = {}
        for h in horizons:
            vals = [e[h] for e in entries if h in e and np.isfinite(e[h])]
            means[h] = float(np.mean(vals)) if vals else float("nan")
        rows.append((cfg, means, len(entries)))

    baseline = next((m for c, m, _ in rows if c == "baseline"), None)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("OrbitalMind — Day-6 Ablation\n")
        fh.write("=" * 78 + "\n\n")
        fh.write("!!! UNITS: these RMSEs are in DIFFERENCED space, the models' own\n")
        fh.write("working space — NOT the reconstructed error in nanoseconds that\n")
        fh.write("run_pipeline.py reports. They are valid for RANKING configurations\n")
        fh.write("against each other and INVALID as absolute accuracy. Do not compare\n")
        fh.write("them to the deck's 0.65 ns or 7.5 ns targets.\n\n")
        fh.write("Scored on the BACKTEST window: the last 24 hours of the 7-day input,\n")
        fh.write("held out from training. The 2026-08-28 holdout file is NOT opened by\n")
        fh.write("this script. Choosing a configuration by its holdout score would be\n")
        fh.write("tuning against the answer, so every number below is validation only.\n\n")
        fh.write(f"Satellites: {', '.join(satellites)}\n")
        fh.write(f"Column:     ClockError_ns\n")
        fh.write(f"Runtime:    {elapsed/60:.1f} min\n")
        if skipped_imf:
            fh.write("NOTE:       noise-IMF pass SKIPPED (--skip-imf)\n")
        fh.write("\n")

        fh.write("RMSE by horizon, averaged across satellites (ns)\n")
        fh.write("-" * 78 + "\n")
        fh.write(f"{'configuration':<22}" + "".join(f"{h:>11}" for h in horizons) + "\n")
        for cfg, means, n in sorted(rows, key=lambda r: r[1].get("1hr", np.inf)):
            fh.write(f"{cfg:<22}" + "".join(f"{means[h]:11.4f}" for h in horizons) + "\n")
        fh.write("\n")

        if baseline:
            fh.write("Change vs baseline at the 1-hour horizon\n")
            fh.write("-" * 78 + "\n")
            b = baseline.get("1hr", float("nan"))
            for cfg, means, n in sorted(rows, key=lambda r: r[1].get("1hr", np.inf)):
                if cfg == "baseline":
                    continue
                d = means.get("1hr", float("nan")) - b
                pct = 100 * d / b if b else float("nan")
                verdict = "worse" if d > 0 else "better"
                fh.write(f"{cfg:<22} {d:+10.4f} ns  ({pct:+6.1f}%)  {verdict}\n")
            fh.write("\n")
            fh.write("How to read the leave-one-out rows: dropping a model that EARNS its\n")
            fh.write("place should make the error WORSE. A 'drop-X' row that is better than\n")
            fh.write("baseline means model X is not contributing on this sample.\n")

        fh.write("\nThis is validation evidence on a small sample, not a final result.\n")
        fh.write("It is a basis for a decision, not a number for the deck.\n")

    print(f"\nWrote {REPORT}")
    for cfg, means, n in sorted(rows, key=lambda r: r[1].get("1hr", np.inf)):
        print(f"  {cfg:<22} 1hr RMSE {means.get('1hr', float('nan')):9.4f} ns  (n={n})")


if __name__ == "__main__":
    main()
