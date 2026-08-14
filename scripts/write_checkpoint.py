#!/usr/bin/env python3
"""
Write a checkpoint after completing a function.
Claude Code calls this after every function it writes and tests.

Usage: python scripts/write_checkpoint.py \
         --iteration 1 \
         --function "generate_synthetic_gnss_data" \
         --status complete \
         --note "generates 6144 rows, seed=42, saved to CSV"
"""
import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument(
        "--status",
        choices=["complete", "in_progress", "failed"],
        required=True,
    )
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    checkpoint_path = f"memory/checkpoints/checkpoint_iter_{args.iteration}.md"

    # Read existing checkpoint
    existing = ""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            existing = f.read()

    # Extract already completed functions
    completed = []
    in_progress = []
    for line in existing.split("\n"):
        if line.startswith("- [x]"):
            completed.append(line.replace("- [x] ", "").strip())
        elif line.startswith("- [ ]"):
            in_progress.append(line.replace("- [ ] ", "").strip())

    # Update function status
    if args.status == "complete":
        if args.function not in completed:
            completed.append(args.function)
        if args.function in in_progress:
            in_progress.remove(args.function)
    elif args.status == "in_progress":
        if args.function not in in_progress and args.function not in completed:
            in_progress.append(args.function)

    # Write updated checkpoint
    content = f"""# Checkpoint: Iteration {args.iteration}
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Last safe point: {args.function} — {args.status}

## Completed functions (tested and verified)
"""
    for fn in completed:
        content += f"- [x] {fn}\n"

    if in_progress:
        content += "\n## In progress (written but not fully tested)\n"
        for fn in in_progress:
            content += f"- [ ] {fn}\n"

    if args.note:
        content += f"\n## Notes\n{args.note}\n"

    content += f"\n## Resume instruction\n"
    if in_progress:
        content += (
            f"Resume from: {in_progress[0]} — complete this function first\n"
        )
    else:
        content += (
            f"All registered functions complete — run make verify-{args.iteration}\n"
        )

    with open(checkpoint_path, "w") as f:
        f.write(content)

    print(
        f"[checkpoint] Iteration {args.iteration}: {args.function} → {args.status}"
    )


if __name__ == "__main__":
    main()
