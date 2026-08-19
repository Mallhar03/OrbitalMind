# OrbitalMind — Architecture

Team Xenith | SH-DST-03 | Smart Horizon 2026

Read this before changing anything in `src/`. It covers the two design
decisions that are not obvious from the code — the **two plans** and the
**two coordinate frames** — plus a map of what is wired in and what is not.

---

## 1. What the pipeline actually does

```
CSV
 │
 ├─ per satellite, per error column (ClockError_ns, EphemerisError_m)
 │   │
 │   ├─ preprocess_satellite()          preprocessing/pipeline.py
 │   │    ├─ remove_outliers_mad()      MAD modified z-score, threshold 3.5
 │   │    ├─ correct_iod_jumps()        adaptive threshold (see §3)
 │   │    ├─ single_difference()        makes the series stationary
 │   │    └─ decompose_signal()         EMD → trend / periodic / noise
 │   │
 │   ├─ combined = trend + periodic     the modelling signal (noise dropped)
 │   │
 │   ├─ compute_splits(len(combined))   splits.py — see §2
 │   │
 │   └─ for each plan (backtest, submission):
 │        ├─ train 4 base models on plan.train
 │        │     LSTM · TCN-LSTM · TFT · Neural ODE
 │        ├─ forecast plan.cal, fit LightGBM meta-learner on cal_meta half
 │        ├─ fit residual flow on cal_flow half     → bias + distribution
 │        ├─ forecast plan.target from plan.input
 │        └─ reconstruct to original units (see §3)
 │
 ├─ outputs/submission.csv              point + sigma + 95% bounds
 ├─ outputs/evaluation_report.txt       RMSE in ns/m vs TWO baselines
 ├─ outputs/shapiro_wilk_result.txt     measured, never manufactured
 └─ outputs/qq_plot.png, residual_histogram.png
```

Run it:

```bash
python src/orbitalmind/run_pipeline.py --data <csv>              # full
python src/orbitalmind/run_pipeline.py --data <csv> --no-backtest  # ~half the time
python src/orbitalmind/run_pipeline.py --data <csv> --max-satellites 2
```

---

## 2. The two plans (`src/orbitalmind/splits.py`)

Every window is derived from each satellite's **actual row count**. Nothing
assumes a fixed length. Let `n` be the differenced series length (rows − 1)
and `h` the 96-step horizon.

| Plan | train | calibration | target |
|------|-------|-------------|--------|
| `backtest` | `[0, n−2h)` | `[n−2h, n−h)` | `[n−h, n)` — **real held-out truth** |
| `submission` | `[0, n−h)` | `[n−h, n)` | `[n, n+h)` — **past the end of the file** |

**Why two.** The deliverable and the measurement are different jobs. The
submission has to forecast day 8, which does not exist inside a 7-day file, so
its target extends past the last row. But you cannot score a window that has
no ground truth, so the backtest holds out the final real day and never lets
training or calibration see it.

This resolves cleanly for every file shape:

| Input | backtest scores | submission forecasts |
|-------|-----------------|----------------------|
| 672 rows (organisers' 7-day format) | day 7 | **day 8** ← the deliverable |
| 768 rows (our synthetic, 8 days) | **day 8** ← scored honestly | day 9 |
| 1344 rows (real IGS, 14 days) | day 14 | day 15 |

The calibration window is split in half: the meta-learner fits on `cal_meta`,
the residual flow on `cal_flow`. The flow therefore sees residuals that are
out-of-sample for the base models **and** for the meta-learner, so the
predictive spread it learns is not optimistic.

> Historical note: these windows were once hardcoded as 480 / 576 / 672. That
> matched only our own 768-row synthetic file. On a 672-row file the "day 8"
> window fell *inside* the data, so the pipeline re-predicted day 7 and
> labelled it day 8. See `memory/what_failed.md`, iteration 9.

---

## 3. The two coordinate frames

`preprocess_satellite()` returns **two** original-scale series. They are not
interchangeable and mixing them up is worth thousands of nanoseconds.

| Key | What it is | Use it for |
|-----|-----------|------------|
| `observed` | outlier-cleaned only — the **measurement frame** | anchoring reconstruction, scoring against truth |
| `original_cleaned` | additionally IOD-corrected — jump-free **modelling frame** | differencing, decomposition, model input |

IOD correction subtracts each detected jump from everything after it, which
slides the series into its own frame. On real IGS data that offset reached
**3339 ns (G02)** and **18216 ns (G03)**. Judges score against the measured
frame, so reconstruction anchors on `observed`.

### Reconstruction

Because `combined[i] ≈ cleaned[i+1] − cleaned[i]`, a forecast of
`combined[t0 : t0+h]` reconstructs as:

```python
original[k] = observed[t0] + cumsum(diff_preds)[k]
```

The anchor is `observed[t0]` — the last value before the window. It was once
`cleaned[0]`, the first sample of the whole record, which carried every
nanosecond of drift since day 1 into the forecast.

### Adaptive IOD threshold

`correct_iod_jumps()` derives its threshold from the robust spread of each
satellite's own first differences:

```
threshold = |median(Δ)| + 8 × 1.4826 × MAD(Δ)
```

A fixed 2.0 ns threshold classified **671 of 671 steps** as IOD jumps for both
G02 (~5 ns/step drift) and G03 (~27 ns/step), and subtracting all of those
offsets removed the entire trend. An IOD upload is a discontinuity relative to
a satellite's *own* behaviour, not to an absolute constant. Pass `threshold_ns`
explicitly to override.

---

## 4. The residual flow — read before touching it

`models/normalizing_flow.py` fits a normalizing flow to out-of-sample
residuals and returns a `ResidualCalibration` carrying:

* `bias` — a **scalar** shift applied to the point forecast
* `sigma` and `sample_pool` — the predictive distribution

`predictive_interval()` turns that into per-point bounds. Because
reconstruction sums `k` differenced steps, spread accumulates as `√k`, so the
interval widens across the horizon.

**The correction must stay a scalar.** An earlier version computed

```python
_correction = (z_gaussian * res_std + res_mean) - residuals   # ← ground truth
corrected   = preds - _correction
```

which made `actual − corrected` a vector of exact normal quantiles *by
definition*. Shapiro-Wilk returned p ≈ 0.9999 for every possible input,
including residuals that were half constant and half exponential. It measured
nothing, and the same ground-truth-derived vector was subtracted from the
day-8 forecast.

`tests/test_normalizing_flow.py` pins this down: two different prediction
vectors must be moved by the identical amount, and deliberately non-Gaussian
residuals must **not** come back with a high p-value.

**Shapiro-Wilk is an outcome, not a gate.** A FAIL means the residuals carry
structure the ensemble has not captured. Fix that in the models or the
features — never in the post-processor.

---

## 5. Module map

| Path | Status |
|------|--------|
| `preprocessing/` | wired in |
| `models/lstm.py`, `tcn_lstm.py`, `tft.py`, `neural_ode.py` | wired in |
| `ensemble/lightgbm_meta.py` | wired in |
| `models/normalizing_flow.py` | wired in |
| `splits.py`, `paths.py` | wired in |
| **`features/`** | **built and tested, NEVER imported by the pipeline** |
| **`models/diffusion.py`** | **built and tested, NEVER imported by the pipeline** |

The proposal credits both orphans: "96 lag features", "FFT spectral features
capturing 12-hour and 24-hour periodicity", and Diffusion as one of the four
ensemble members. Today the models run **univariate** on `combined`, and the
fourth member is a plain LSTM, which the proposal never mentions. Closing this
gap is open work.

Also open: all four models use `nn.MSELoss()`, not the Gaussian likelihood
loss the proposal claims, and `device="cpu"` is hardcoded in every train and
predict signature, so nothing uses a GPU.

---

## 6. Conventions

* All randomness seeded with 42.
* Paths resolve from the repo root via `paths.py` — the pipeline runs from any
  working directory.
* Every function carries a docstring with Args and Returns.
* Models train on **exactly** the array they are handed. The caller owns the
  window; no function slices its own training data.
* Failures never write placeholder zeros. A failed satellite falls back to
  persistence and is listed in `evaluation_report.txt`.

---

## 7. Where the history lives

* `memory/what_failed.md` — the seven defects found in iteration 9, each with
  root cause, measurement, and fix. Read this before trusting any older claim.
* `memory/decisions.md` — architecture decisions and rejected alternatives.
* `memory/never_do.md` — hard constraints.
* `skills/*.md` — per-module specifications.
