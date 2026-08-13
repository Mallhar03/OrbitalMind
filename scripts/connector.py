#!/usr/bin/env python3
"""
OrbitalMind Connector Script
Called by Makefile after each iteration completes.
Acts automatically — does not suggest.

Usage: python scripts/connector.py --iteration N --status pass|fail --rmse 1.23
"""
import argparse
import subprocess
import os
import re
from datetime import datetime


ITERATION_MODULES = {
    1: "Synthetic Data Generator",
    2: "Preprocessing Pipeline",
    3: "Feature Engineering",
    4: "LSTM + TCN-LSTM + Neural ODE",
    5: "TFT Model",
    6: "LightGBM Meta-Learner",
    7: "Normalizing Flow",
    8: "Full Pipeline Integration",
}


def update_readme_status(iteration: int, status: str):
    """Update the build status table in README.md."""
    readme_path = "README.md"
    with open(readme_path, "r") as f:
        content = f.read()

    module = ITERATION_MODULES.get(iteration, f"Iteration {iteration}")
    old_line = f"| {iteration} | {module} | ⬜ Not started |"
    new_line = f"| {iteration} | {module} | {'✅ Complete' if status == 'pass' else '❌ Failed'} |"

    if old_line in content:
        content = content.replace(old_line, new_line)
    else:
        in_progress = f"| {iteration} | {module} | 🔄 In progress |"
        if in_progress in content:
            content = content.replace(in_progress, new_line)

    with open(readme_path, "w") as f:
        f.write(content)
    print(f"[connector] README updated: Iteration {iteration} → {status}")


def update_current_iteration(iteration: int, status: str, rmse: float = None):
    """Update memory/current_iteration.md."""
    path = "memory/current_iteration.md"
    next_iter = iteration + 1

    content = f"""# Current Iteration

Number: {iteration if status == 'fail' else next_iter}
Status: {'FAILED — awaiting fix' if status == 'fail' else 'IN PROGRESS'}
Module: {ITERATION_MODULES.get(next_iter, 'All complete') if status == 'pass' else ITERATION_MODULES.get(iteration)}
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Previous iteration result
Iteration {iteration}: {'PASS' if status == 'pass' else 'FAIL'}
{'RMSE at 1hr: ' + str(rmse) + ' ns' if rmse else ''}
Checker verdict: {'APPROVED' if status == 'pass' else 'PENDING FIX'}

## Next action
{'Build ' + ITERATION_MODULES.get(next_iter, 'Submit v1') if status == 'pass' else 'Fix failures listed in memory/checker_verdict.md'}
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"[connector] current_iteration.md updated")


def update_what_is_done(iteration: int, rmse: float = None):
    """Append completed iteration to memory/what_is_done.md."""
    path = "memory/what_is_done.md"
    with open(path, "r") as f:
        content = f.read()

    entry = f"""
---
Iteration: {iteration}
Module: {ITERATION_MODULES.get(iteration)}
File: src/orbitalmind/
Test: tests/test_{get_test_name(iteration)}.py
Metric achieved: {'RMSE ' + str(rmse) + ' ns at 1hr' if rmse else 'pipeline runs clean'}
Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Checker verdict: PASS
---
"""
    with open(path, "a") as f:
        f.write(entry)
    print(f"[connector] what_is_done.md updated")


def git_commit(iteration: int, rmse: float = None):
    """Auto git commit after iteration passes."""
    module = ITERATION_MODULES.get(iteration, f"Iteration {iteration}")
    rmse_str = f" — RMSE {rmse:.3f}ns" if rmse else ""
    message = f"Iteration {iteration} PASS: {module}{rmse_str}"

    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    print(f"[connector] Git committed: {message}")


def get_test_name(iteration: int) -> str:
    """Return test file base name for the given iteration."""
    names = {
        1: "synthetic_data",
        2: "preprocessing",
        3: "features",
        4: "lstm",
        5: "tft",
        6: "meta_learner",
        7: "normalizing_flow",
        8: "pipeline",
    }
    return names.get(iteration, f"iteration_{iteration}")


def main():
    parser = argparse.ArgumentParser(description="OrbitalMind iteration connector")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--status", choices=["pass", "fail"], required=True)
    parser.add_argument("--rmse", type=float, default=None)
    args = parser.parse_args()

    print(f"\n[connector] Running for Iteration {args.iteration} — {args.status.upper()}")

    if args.status == "pass":
        update_readme_status(args.iteration, "pass")
        update_what_is_done(args.iteration, args.rmse)
        update_current_iteration(args.iteration, "pass", args.rmse)
        git_commit(args.iteration, args.rmse)
        print(f"[connector] ✓ All actions complete for Iteration {args.iteration}")
    else:
        update_readme_status(args.iteration, "fail")
        update_current_iteration(args.iteration, "fail", args.rmse)
        print(f"[connector] ✗ Iteration {args.iteration} marked failed")

    print("[connector] Done.\n")


if __name__ == "__main__":
    main()
