# OrbitalMind
### Physics-Aware Ensemble AI for GNSS Clock & Ephemeris Error Prediction
**Team Xenith | SH-DST-03 | Smart Horizon 2026**

---

**New here? Read [ARCHITECTURE.md](ARCHITECTURE.md) first.** It explains the
two window plans and the two coordinate frames — the parts of `run_pipeline.py`
that are not obvious from reading the code, and the parts most likely to be
broken by a well-meaning change.

---

## Quick Start

Requires **Python 3.10–3.12**. Check with `python3 --version`.

```bash
git clone <this repo> && cd OrbitalMind

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# smoke test — 2 satellites, ~5 min, confirms the install works
python src/orbitalmind/run_pipeline.py \
    --data data/synthetic/gnss_synthetic.csv --max-satellites 2

pytest tests/ --ignore=tests/test_pipeline.py -q     # ~3 min, 102 tests
```

`tests/test_pipeline.py` runs the whole pipeline and takes ~40 minutes. Run it
on its own when you actually want end-to-end verification.

### Getting the real data

`data/raw/` is gitignored, so a fresh clone has only the synthetic file. To
download real IGS orbit and clock data from NASA CDDIS:

```bash
python scripts/fetch_data.py        # writes data/raw/gnss_real.csv
```

This needs CDDIS credentials in `~/.netrc` (free registration at
`urs.earthdata.nasa.gov`). Note that `fetch_data.py` currently falls back to
synthetic data *silently* on any download failure — check that
`data/raw/gnss_real.csv` actually contains real satellite IDs (G01, G02, …)
and not GEO-01/MEO-01 before trusting a run.

### Troubleshooting

| Symptom | Cause |
|---------|-------|
| `ImportError: orbitalmind.splits` | stale checkout — `git pull` |
| `pip install` fails on torch | Python 3.13+ is unsupported; use 3.10–3.12 |
| `ValueError: record too short` | fewer than 385 rows for a satellite; the split needs seq_len + 3×horizon |
| `ValueError: unrecognised OrbitType` | the CSV's OrbitType column holds something other than GEO/GSO/IGSO/MEO |
| Run is very slow | expected — see the runtime note below; use `--no-backtest` or `--max-satellites` |

**Runtime.** Roughly 5 min per satellite on CPU with the backtest enabled
(each satellite trains four models twice). A 31-satellite record is a
multi-hour job. `device="cpu"` is hardcoded in every model, so no GPU is used
yet.

---

## Build Status

| Iteration | Module | Built | Wired into pipeline |
|-----------|--------|-------|---------------------|
| 1 | Synthetic Data Generator | yes | yes (see caveat below) |
| 2 | Preprocessing Pipeline | yes | yes |
| 3 | Feature Engineering | yes | **NO — orphaned** |
| 4 | LSTM + TCN-LSTM + Neural ODE | yes | yes |
| 5 | TFT Model | yes | yes |
| 6 | LightGBM Meta-Learner | yes | yes |
| 7 | Normalizing Flow | yes | yes (rebuilt, iteration 9) |
| 8 | Full Pipeline Integration | yes | yes |
| - | Diffusion model | yes | **NO — orphaned** |

`src/orbitalmind/features/` and `src/orbitalmind/models/diffusion.py` are
implemented and tested but never imported by `run_pipeline.py`. The proposal
credits both. The models currently run univariate on the combined signal, so
no lag, rolling or FFT feature reaches any of them.

---

## Measured Results (synthetic data, 8 satellites, held-out final 24 h)

These come from the backtest plan: the scored day is never seen during
training or calibration, and RMSE is in original units, not in differenced
EMD-filtered space.

| Metric | Result |
|--------|--------|
| Clock RMSE @ 1 hr | 0.17 – 0.74 ns across 8 satellites |
| Beat persistence baseline @ 1 hr | **5 of 16** satellite/error combinations |
| Shapiro-Wilk on held-out residuals | **FAIL** (p < 1e-6, n = 1536) |
| Fallbacks used | 0 |
| Runtime | ~40 min, 8 satellites, CPU, with backtest |

The clock figures sit near the 0.65 ns proposal target, but the ensemble
loses to carrying the last observed value forward on 7 of 8 clock series.
The absolute number is small because the signal is easy, not because the
model is good. Closing that gap is the open work.

### Known data caveat
`synthetic_generator.py` clips EphemerisError_m to ±5 m. GEO satellites
saturate at exactly 5.000 and stay pinned there for the whole held-out day,
so persistence scores 0.000 and the model scores up to 65 m against it. The
three GEO ephemeris rows in the evaluation report measure a clipping
artifact, not forecasting skill.

---

## Verify Conditions
- `pytest tests/` runs with zero failures
- Clock RMSE at 1 hr beats the persistence baseline on a majority of
  satellites, reported in ns against held-out truth
- Shapiro-Wilk p is **reported** on held-out residuals — it is an outcome,
  never a gate to be engineered. See `skills/normalizing_flow.md`.
- Full pipeline runs in one command

## Pipeline flags
```bash
# full run with honest scoring (slow: trains each satellite twice)
python src/orbitalmind/run_pipeline.py --data data/synthetic/gnss_synthetic.csv

# submission only, roughly half the runtime
python src/orbitalmind/run_pipeline.py --data <csv> --no-backtest

# smoke test on the first 2 satellites
python src/orbitalmind/run_pipeline.py --data <csv> --max-satellites 2
```

Windows are derived from each satellite's actual row count, so a 7-day
(672-row) file, the 8-day synthetic file, and a 14-day real record all work.
The submission forecast starts *after* the last row of the input.
