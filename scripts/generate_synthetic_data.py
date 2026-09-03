#!/usr/bin/env python3
"""Generate the local synthetic GNSS dataset used for the install smoke test.

Not committed to git (see data/README.md) — run this once after `make setup`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-geo", type=int, default=3, help="number of GEO satellites")
    parser.add_argument("--n-meo", type=int, default=5, help="number of MEO satellites")
    parser.add_argument("--n-days", type=int, default=8, help="total days to generate")
    parser.add_argument("--interval-minutes", type=int, default=15, help="sampling interval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path", default="data/synthetic/gnss_synthetic.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    generate_synthetic_gnss_data(
        n_geo=args.n_geo,
        n_meo=args.n_meo,
        n_days=args.n_days,
        interval_minutes=args.interval_minutes,
        seed=args.seed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
