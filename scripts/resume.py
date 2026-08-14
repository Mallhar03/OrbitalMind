#!/usr/bin/env python3
"""
OrbitalMind Resume Script
Run this at the START of every Claude Code session.
Diagnoses exact project state and tells Claude Code where to resume.

Usage: python scripts/resume.py
"""
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime


def read_file(path: str) -> str:
    """Read file contents, return empty string if not found."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def check_file_complete(filepath: str, required_functions: list) -> dict:
    """Check which functions exist in a file."""
    content = read_file(filepath)
    results = {}
    for func in required_functions:
        results[func] = f"def {func}" in content
    return results


def run_tests_safe(test_file: str) -> str:
    """Run a test file and return pass/fail without crashing."""
    if not os.path.exists(test_file):
        return "NO_TEST_FILE"
    result = subprocess.run(
        ["python3", "-m", "pytest", test_file, "-q", "--tb=no"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        return "PASS"
    elif "no tests ran" in result.stdout:
        return "NO_TESTS"
    else:
        return f"FAIL: {result.stdout[-200:]}"


def get_current_iteration() -> dict:
    """Read current iteration from memory."""
    content = read_file("memory/current_iteration.md")
    number_match = re.search(r"Number:\s*(\d+)", content)
    status_match = re.search(r"Status:\s*(.+)", content)
    return {
        "number": int(number_match.group(1)) if number_match else 0,
        "status": status_match.group(1).strip() if status_match else "UNKNOWN"
    }


def get_checker_verdict(iteration: int) -> str:
    """Check if checker approved this iteration."""
    content = read_file("memory/checker_verdict.md")
    pattern = f"Iteration: {iteration}.*?VERDICT: (PASS|FAIL)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    return "NO_VERDICT"


ITERATION_TESTS = {
    1: ("tests/test_synthetic_data.py", []),
    2: ("tests/test_preprocessing.py", []),
    3: ("tests/test_features.py", []),
    4: ("tests/test_lstm.py", []),
    5: ("tests/test_tft.py", []),
    6: ("tests/test_meta_learner.py", []),
    7: ("tests/test_normalizing_flow.py", []),
    8: ("tests/test_pipeline.py", []),
}

ITERATION_FILES = {
    1: ["src/orbitalmind/utils/synthetic_generator.py"],
    2: [
        "src/orbitalmind/preprocessing/outlier_removal.py",
        "src/orbitalmind/preprocessing/iod_correction.py",
        "src/orbitalmind/preprocessing/differencing.py",
        "src/orbitalmind/preprocessing/decomposition.py",
        "src/orbitalmind/preprocessing/pipeline.py",
    ],
    3: [
        "src/orbitalmind/features/lag_features.py",
        "src/orbitalmind/features/rolling_features.py",
        "src/orbitalmind/features/fft_features.py",
        "src/orbitalmind/features/feature_matrix.py",
    ],
    4: [
        "src/orbitalmind/models/lstm.py",
        "src/orbitalmind/models/tcn_lstm.py",
        "src/orbitalmind/models/neural_ode.py",
        "src/orbitalmind/models/diffusion.py",
        "src/orbitalmind/models/base_trainer.py",
    ],
    5: ["src/orbitalmind/models/tft.py"],
    6: ["src/orbitalmind/ensemble/lightgbm_meta.py"],
    7: ["src/orbitalmind/models/normalizing_flow.py"],
    8: ["src/orbitalmind/run_pipeline.py"],
}


def get_skill_name(iteration: int) -> str:
    """Return skill file name for the given iteration."""
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
    print("\n" + "=" * 60)
    print("ORBITALMIND — SESSION RESUME DIAGNOSTICS")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Step 1: Current iteration state ──
    current = get_current_iteration()
    iter_num = current["number"]
    iter_status = current["status"]

    print(f"\n[1] CURRENT ITERATION: {iter_num}")
    print(f"    Status: {iter_status}")

    # ── Step 2: Check checkpoint ──
    checkpoint_path = f"memory/checkpoints/checkpoint_iter_{iter_num}.md"
    checkpoint = read_file(checkpoint_path)
    if checkpoint:
        last_safe = re.search(r"Last safe point:\s*(.+)", checkpoint)
        completed_fns = re.search(r"Completed functions:\s*(.+)", checkpoint)
        print(f"\n[2] CHECKPOINT FOR ITERATION {iter_num}:")
        print(f"    Last safe point: {last_safe.group(1) if last_safe else 'NOT FOUND'}")
        print(f"    Completed functions: {completed_fns.group(1) if completed_fns else 'NONE'}")
    else:
        print(f"\n[2] NO CHECKPOINT FOUND for iteration {iter_num}")
        print(f"    → Start iteration {iter_num} from the beginning")

    # ── Step 3: Check which files exist ──
    print(f"\n[3] FILE STATUS FOR ITERATION {iter_num}:")
    files = ITERATION_FILES.get(iter_num, [])
    missing_files = []
    for f in files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        status = f"EXISTS ({size} bytes)" if exists else "MISSING"
        print(f"    {'✓' if exists else '✗'} {f} — {status}")
        if not exists:
            missing_files.append(f)

    # ── Step 4: Run tests for current iteration ──
    print(f"\n[4] TEST STATUS:")
    test_file, _ = ITERATION_TESTS.get(iter_num, ("", []))
    test_result = "NO_TEST_FILE"
    if test_file:
        test_result = run_tests_safe(test_file)
        print(f"    {test_file}: {test_result}")
    else:
        print(f"    No test file registered for iteration {iter_num}")

    # ── Step 5: Check checker verdict ──
    print(f"\n[5] CHECKER VERDICT FOR ITERATION {iter_num}:")
    verdict = get_checker_verdict(iter_num)
    print(f"    {verdict}")

    # ── Step 6: Check git status ──
    print(f"\n[6] GIT STATUS:")
    git_result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True
    )
    uncommitted = git_result.stdout.strip()
    if uncommitted:
        print(f"    Uncommitted changes:\n{uncommitted}")
    else:
        print(f"    Working tree clean")

    git_log = subprocess.run(
        ["git", "log", "--oneline", "-3"],
        capture_output=True, text=True
    )
    print(f"    Last 3 commits:\n{git_log.stdout}")

    # ── Step 7: RESUME INSTRUCTION ──
    print("\n" + "=" * 60)
    print("RESUME INSTRUCTION FOR CLAUDE CODE")
    print("=" * 60)

    if verdict == "PASS" and "CHECKER APPROVED" in iter_status:
        next_iter = iter_num + 1
        if next_iter > 8:
            print("✅ ALL ITERATIONS COMPLETE. v1 is done.")
        else:
            print(f"✅ Iteration {iter_num} is fully complete and checker approved.")
            print(f"→ START: Iteration {next_iter}")
            print(f"→ READ:  skills/{get_skill_name(next_iter)}.md")
            print(f"→ WRITE: checkpoint_iter_{next_iter}.md before first function")

    elif test_file and test_result == "PASS" and verdict == "NO_VERDICT":
        print(f"⚠️  Iteration {iter_num}: Tests PASS but checker not called yet.")
        print(f"→ DO NOT write more code")
        print(f"→ RUN: make request-review")
        print(f"→ Open a NEW session with .claude/checker/CHECKER.md")

    elif missing_files:
        print(f"⚠️  Iteration {iter_num}: {len(missing_files)} files missing.")
        print(f"→ RESUME writing from checkpoint last safe point")
        print(f"→ Missing files to create:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"→ READ checkpoint: {checkpoint_path}")
        print(f"→ READ skill: skills/{get_skill_name(iter_num)}.md")

    elif test_file and "FAIL" in test_result:
        print(f"❌ Iteration {iter_num}: Files exist but tests FAIL.")
        print(f"→ READ: memory/what_failed.md for last failure")
        print(f"→ READ: memory/checker_verdict.md for rejection details")
        print(f"→ FIX: the single root cause identified in what_failed.md")
        print(f"→ RUN: make verify-{iter_num}")

    else:
        print(f"⚠️  Iteration {iter_num}: Status unclear.")
        print(f"→ READ memory/current_iteration.md carefully")
        print(f"→ READ memory/what_failed.md")
        print(f"→ READ checkpoint: {checkpoint_path}")
        print(f"→ Do NOT assume anything — verify each file manually")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
