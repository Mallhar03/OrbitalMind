"""
Model training and ranking script — Person 3 lane.

Trains HarmonicPredictor and GaussianProcessPredictor on all five training
series and scores them against matched test series using the canonical
score_residuals() scorer.

Usage:
    source venv/bin/activate
    PYTHONPATH=src python3 scripts/train_and_rank.py

Output:
    - Ranking table printed to stdout
    - Per-series results saved to outputs/ranking_results.json
    - Best model per series saved via pickle to models/saved/best_<series_id>.pkl
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from orbitalmind.ingest import load_dataset, load_series
from orbitalmind.interfaces import PersistencePredictor, residuals_by_parameter
from orbitalmind.evaluation.scoring import score_residuals
from orbitalmind.models.harmonic import HarmonicPredictor
from orbitalmind.models.gp_predictor import GaussianProcessPredictor

# ── Data paths ────────────────────────────────────────────────────────────────

TRAIN_FILES = [
    "data/DATA_GEO_Train.csv",
    "data/DATA_MEO_Train.csv",
    "data/DATA_MEO_Train2.csv",
]
TEST_FILES = [
    "data/DATA_GEO_Test.csv",
    "data/DATA_MEO_Test.csv",
    "data/DATA_MEO_Test2.csv",
]


def _match_test_to_train(train_series, test_series):
    """
    Pair each training series with its matching test series.

    Matching rule: same orbit type + same block index (block 0 of GEO_Train
    matches block 0 of GEO_Test, etc.).
    """
    pairs = []
    for tr in train_series:
        for te in test_series:
            if tr.orbit == te.orbit and tr.block == te.block:
                pairs.append((tr, te))
                break
    return pairs


def _build_persistence(train: "Series") -> PersistencePredictor:
    """Build a persistence predictor from the last row of a training series."""
    last_row = train.values()[-1]
    return PersistencePredictor(last_values=last_row)


def train_and_rank():
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Loading training and test series …")
    train_all = load_dataset(TRAIN_FILES)
    test_all  = load_dataset(TEST_FILES)

    pairs = _match_test_to_train(train_all, test_all)
    print(f"Matched {len(pairs)} train/test series pairs.\n")

    all_results = {}

    header = f"{'Series':<20} {'Model':<22} {'W':>8} {'p':>8} {'H':>4}  {'beats_persistence':>18}"
    print(header)
    print("─" * len(header))

    for train, test in pairs:
        sid = train.satellite_id
        results_for_sid = {}

        # ── Persistence baseline ──────────────────────────────────────────
        t0 = time.time()
        pers = _build_persistence(train)
        resids_p = residuals_by_parameter(pers, test)
        score_p  = score_residuals(resids_p)
        results_for_sid["Persistence"] = {
            "W": score_p.W, "p": score_p.p_value, "H": score_p.H,
            "per_parameter": {k: {"W": v.W, "mean": v.mean, "std": v.std}
                              for k, v in score_p.per_parameter.items()},
            "time_s": round(time.time() - t0, 3),
        }
        print(f"{sid:<20} {'Persistence':<22} {score_p.W:>8.4f} {score_p.p_value:>8.4f} {score_p.H:>4}  {'—':>18}")

        # ── Harmonic ─────────────────────────────────────────────────────
        t0 = time.time()
        har = HarmonicPredictor().fit(train)
        resids_h = residuals_by_parameter(har, test)
        score_h  = score_residuals(resids_h)
        beats_h  = "✓ YES" if score_h.W > score_p.W else "✗ NO"
        results_for_sid["Harmonic"] = {
            "W": score_h.W, "p": score_h.p_value, "H": score_h.H,
            "beats_persistence": score_h.W > score_p.W,
            "per_parameter": {k: {"W": v.W, "mean": v.mean, "std": v.std}
                              for k, v in score_h.per_parameter.items()},
            "time_s": round(time.time() - t0, 3),
        }
        print(f"{sid:<20} {'Harmonic':<22} {score_h.W:>8.4f} {score_h.p_value:>8.4f} {score_h.H:>4}  {beats_h:>18}")

        # ── Gaussian Process ──────────────────────────────────────────────
        print(f"{sid:<20} {'GP (fitting…)':<22}", end="", flush=True)
        t0 = time.time()
        gp  = GaussianProcessPredictor(n_restarts=2).fit(train)
        resids_g = residuals_by_parameter(gp, test)
        score_g  = score_residuals(resids_g)
        beats_g  = "✓ YES" if score_g.W > score_p.W else "✗ NO"
        elapsed  = round(time.time() - t0, 1)
        results_for_sid["GP"] = {
            "W": score_g.W, "p": score_g.p_value, "H": score_g.H,
            "beats_persistence": score_g.W > score_p.W,
            "per_parameter": {k: {"W": v.W, "mean": v.mean, "std": v.std}
                              for k, v in score_g.per_parameter.items()},
            "time_s": elapsed,
        }
        print(f"\r{sid:<20} {'GP':<22} {score_g.W:>8.4f} {score_g.p_value:>8.4f} {score_g.H:>4}  {beats_g:>18}  ({elapsed}s)")

        # ── Pick best model and save ──────────────────────────────────────
        candidates = {"Harmonic": (har, score_h.W), "GP": (gp, score_g.W)}
        best_name, (best_model, best_W) = max(candidates.items(), key=lambda x: x[1][1])
        results_for_sid["best"] = best_name
        results_for_sid["best_W"] = best_W

        save_path = f"models/saved/best_{sid}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"  → Best: {best_name} (W={best_W:.4f})  saved to {save_path}\n")

        all_results[sid] = results_for_sid

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SUMMARY — Best model per series")
    print("═" * 60)
    for sid, res in all_results.items():
        print(f"  {sid:<20} → {res['best']:<12} W={res['best_W']:.4f}")

    all_W = [res["best_W"] for res in all_results.values()]
    pers_W = [res["Persistence"]["W"] for res in all_results.values()]
    print(f"\n  Mean best-model W   : {np.mean(all_W):.4f}")
    print(f"  Mean persistence W  : {np.mean(pers_W):.4f}")
    print(f"  Net W gain          : {np.mean(all_W) - np.mean(pers_W):+.4f}")

    # ── Persist results ───────────────────────────────────────────────────────
    with open("outputs/ranking_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results saved to outputs/ranking_results.json")
    print("Training complete ✓")


if __name__ == "__main__":
    train_and_rank()
