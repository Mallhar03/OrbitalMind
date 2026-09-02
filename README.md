# OrbitalMind
### Physics-Aware Ensemble AI for GNSS Clock & Ephemeris Error Prediction
**Team Xenith | SH-DST-03 | Smart Horizon 2026**

---

GNSS satellites broadcast a predicted clock offset and orbital position. Both are
slightly wrong, and the error drifts in structured ways. OrbitalMind forecasts the
next 24 hours of that error at 15-minute intervals, per satellite, from seven days
of history — so a receiver can correct for it in advance.

## Where the project actually stands

Scored on **2026-08-28**, a day held out entirely: never trained on, never tuned
against, opened once after the forecast was frozen. 95 satellites, 9,120 forecasts.

**It works up to about an hour, and degrades badly beyond two.**

| Horizon | Clock (ns) | vs persistence | Position (m) | vs persistence |
|---------|-----------|----------------|--------------|----------------|
| 15 min  | **0.775** | beats it | 0.094 | level |
| 30 min  | **0.778** | beats it | **0.100** | beats it |
| 1 hr    | **0.938** | beats it | **0.134** | beats it |
| 2 hr    | 1.177 | loses | **0.170** | beats it |
| 12 hr   | 2.842 | loses badly | 0.353 | loses badly |
| 24 hr   | 5.849 | loses badly | 0.786 | loses badly |

Persistence means "repeat the last observed value". At 24 hours it scores 1.62 ns
on clock and 0.13 m on position — so doing nothing beats this ensemble by 3.6x and
6x respectively at that range.

**Against the proposal's targets:** the 1-hour clock target of 0.65 ns is **not
met** (0.938 ns). The 24-hour target of 7.5 ns is met numerically (5.85 ns), but
that figure should not be presented as a success while a trivial baseline scores
1.62 ns.

### Why long horizons fail

The pipeline predicts 96 successive step-changes and accumulates them from an
anchor. Small per-step errors compound: by step 96 they dominate the forecast.
Short horizons involve few accumulations and the models' skill shows through; long
horizons are dominated by accumulated error.

This was measured, not assumed. A scalar bias correction that was being added to
all 96 steps before accumulation turned out to be statistically indistinguishable
from zero — a median of ~48 residuals whose standard error was the same size as
itself. Removing it improved position accuracy by 16% at 24 hours and raised the
count of satellites beating a linear baseline from 145/475 to 183/475. The
amplifying structure remains and is the main open problem.

### What is honest to claim

- Sub-nanosecond clock error at 15 and 30 minutes, beating both baselines
- Real skill at horizons up to one hour, on both error types
- Residual normality is **measured and reported, never engineered** — the
  Shapiro-Wilk test currently fails (p = 0.000000), which is a real finding about
  the ensemble, not a defect to hide
- Every satellite gets its own fitted models; no satellite's dynamics are averaged
  into another's

### What is not

- The 0.65 ns target at 1 hour
- Any claim of skill beyond two hours
- FFT spectral features reaching a model — the periodicity information supplied to
  the ensemble is explicit 12h/24h encodings, not FFT amplitudes

Slide wording that needs correcting before the deck is presented is listed in
[`.claude/audit/DECK-CORRECTIONS.md`](.claude/audit/DECK-CORRECTIONS.md).

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

pytest tests/ --ignore=tests/test_pipeline.py -q     # ~3 min, 161 tests
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
`urs.earthdata.nasa.gov`).

`fetch_data.py` fails loudly by default rather than substituting synthetic
data — a silent substitution writes synthetic rows to the same path with the
same columns a real pull would use, which is indistinguishable downstream. Pass
`--allow-synthetic` to opt into the fallback deliberately.

Every pull also writes `data/raw/gnss_real.provenance.json` recording the source
products, archive, date spans, counts and error definitions, with
`"origin": "real"` or `"origin": "synthetic"`. Check that file to know what a
dataset actually is.

### Troubleshooting

| Symptom | Cause |
|---------|-------|
| `ImportError: orbitalmind.splits` | stale checkout — `git pull` |
| `pip install` fails on torch | Python 3.13+ is unsupported; use 3.10–3.12 |
| `ValueError: record too short` | fewer than 385 rows for a satellite; the split needs seq_len + 3×horizon |
| `ValueError: unrecognised OrbitType` | the CSV's OrbitType column holds something other than GEO/GSO/IGSO/MEO |
| Run is very slow | expected — see the runtime note below; use `--no-backtest` or `--max-satellites` |

**Runtime.** Roughly 5 min per satellite on CPU with the backtest enabled
(each satellite trains four models twice), so a full record is a multi-hour job.

Device selection is resolved through `orbitalmind.device.resolve_device()`:
every model takes `device=None` meaning "decide for me", and the choice is
CUDA, then Apple MPS, then CPU, overridable with the `ORBITALMIND_DEVICE`
environment variable. Expect roughly 2-3x from a GPU rather than 20x — these
models are small and the cost is dominated by the 96-step autoregressive
rollout and the Neural ODE's rk4 solve, both sequential and latency-bound.

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
| - | Diffusion model | yes | **NO — see below** |
| - | Engineered features to meta-learner | yes | yes, behind `--features` |
| - | GPU device selection | yes | yes (`device.py`, all four models) |

`src/orbitalmind/features/` and `src/orbitalmind/models/diffusion.py` are
implemented and tested but never imported by `run_pipeline.py`.

**Diffusion** is not wired in deliberately: `train_diffusion()` fits an
unconditional Gaussian to pooled residuals and takes no input sequence, so
connecting it as-is would feed the meta-learner noise rather than a forecast.
Making it conditional is a redesign. The ensemble that actually runs is
**LSTM + TCN-LSTM + TFT + Neural ODE**.

**features/** is superseded rather than pending. Lag and rolling features of the
forecast horizon cannot be computed at forecast time — they would require the
values being predicted. What reaches the meta-learner under `--features` is
`run_pipeline._meta_features()`: explicit 12-hour and 24-hour periodicity
encodings and the horizon step. FFT amplitudes were tried there and measured
exactly zero gain, because they describe the history the whole forecast is made
from and so cannot vary across it.

---

## Backtest results — how the models were developed

The numbers at the top of this file are the held-out day-8 score, which is the one
that counts. During development the pipeline was tuned against a **backtest
window**: the final 24 hours of the 7-day input, held out from training. Those
figures are recorded here because they are what the design decisions were made on,
and because the gap between them and the day-8 result is itself informative.

Backtest, 95 satellites, features on, engineered features enabled:

| Horizon | Clock RMSE | Beats linear | Position RMSE | Beats linear |
|---------|-----------|--------------|---------------|--------------|
| 15 min  | 1.3153 ns | 82 / 95 | 0.0307 m | 37 / 95 |
| 1 hr    | 1.2279 ns | 74 / 95 | 0.0510 m | 39 / 95 |
| 24 hr   | 2.2015 ns | 31 / 95 | 0.2784 m | 19 / 95 |

**The backtest was optimistic at long range.** It predicted 2.20 ns for clock at 24
hours; the real day-8 result was 5.85 ns. Short horizons went the other way — the
backtest predicted 1.23 ns at 1 hour and the real result was 0.94 ns. Treat
backtest figures as directional, not as a promise.

### Which components earn their place

From `outputs/ablation_report.txt`, produced by `scripts/ablation.py` — 8
satellites, validation window only, never the held-out day. These figures are in
differenced space and rank configurations against each other; they are **not**
comparable to the targets above.

| Change | Effect at 1 hr |
|--------|----------------|
| Drop LSTM | 2.9% worse |
| Drop TCN-LSTM | 2.7% worse |
| Drop TFT | 0.5% worse |
| Drop Neural ODE | 0.9% better (marginal) |
| Add back the discarded noise IMF | **29.8% worse** |
| Engineered features on | **2.2% better** |

All four base models are retained. Three of the four make the forecast worse when
removed; the Neural ODE's contribution is marginal, which suggests it is underfed
rather than useless. The high-frequency component discarded during decomposition
should stay discarded — restoring it is substantially worse.

### Approaches tested and rejected

Recorded so they are not retried:

- **Linear baseline as a meta-learner feature — impossible.** A linear
  extrapolation is a constant slope, so in differenced space it is a zero-variance
  column, and a tree model cannot split on it.
- **Predicting the level directly instead of differencing — fails its premise.**
  With the same model class, differencing wins on 17 of 25 satellites.
- **Shrinking the bias correction rather than removing it — worse.** Removing it
  entirely beat shrinking it on 21–23 of 30 satellites across horizons.

### Resolved: the synthetic clipping caveat
`synthetic_generator.py` used to clip EphemerisError_m to ±5 m, saturating GEO
satellites at exactly 5.000 for the whole held-out day — 801 rows, 34.8% of all
GEO data. Persistence then scored 0.000 and the model scored up to 65 m against
it, so those rows measured a clipping artifact rather than forecasting skill.

**The clipping is gone** and the generator no longer saturates. Any evaluation
report produced before 2026-08-31 was run against the clipped data and its GEO
ephemeris rows should not be trusted.

---

## Verify Conditions
- `pytest tests/` runs with zero failures
- Clock RMSE at 1 hr beats the **linear extrapolation** baseline on a majority of
  satellites, reported in ns against held-out truth. Persistence alone is too weak
  a baseline for a drifting clock, so the report prints both. **Met** — 74/95 on
  the backtest, and the day-8 result beats both baselines at 1 hour.
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
