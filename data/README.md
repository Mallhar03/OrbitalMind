# data/

Nothing under this directory is committed to git — every file here is either
downloaded or generated locally. This file (and the layout it documents) is
the only thing tracked.

| Path | How it gets there | Command |
|------|--------------------|---------|
| `data/raw/gnss_real.csv` (+ `.provenance.json`) | Downloaded from NASA CDDIS / GFZ | `make fetch` (needs `~/.netrc` CDDIS credentials) |
| `data/raw/gnss_holdout.csv` | Written alongside the real pull — day 8, never trained on | `make fetch` |
| `data/synthetic/gnss_synthetic.csv` | Generated locally for the install smoke test and unit tests | `make synthetic` |
| Your own dataset (e.g. a `Data_PS-08`-style export) | Copy it in yourself | — |

Bringing your own dataset in as `data/<name>/...` works as long as it matches
the CSV schema `run_pipeline.py` expects (see `ARCHITECTURE.md` §1) — the
pipeline reads whatever path you pass via `--data`, it does not require the
file to live under `raw/` or `synthetic/`.

Why nothing here is committed: a committed data file can silently drift out of
sync with the code that generates or fetches it — this repo had exactly that
problem (a committed `gnss_synthetic.csv` went missing from git without the
docs or Makefile changing to match, breaking the quickstart for anyone who
didn't notice). Regenerating on demand keeps the data and the code that makes
it in sync by construction, and keeps the repo free of files that are either
large (real pulls) or someone else's to own (your own dataset).
