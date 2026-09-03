# OrbitalMind — Complete Solution Report

Generated 2026-09-03 (updated). Everything measured/verified here was run against
the real code and real files in this session — nothing is quoted from the deck
without a "measured" or "claimed" label telling you which it is.

**Repo:** `Mallhar03/OrbitalMind` · **Working branch for this report's changes:**
`feature/inference-entrypoint-and-shaping` (pushed, no PR yet) · **Base:** `main`

---

## 0. TL;DR

- The repo went through a full audit and cleanup this session: a duplicated
  project tree was found and removed (PR #4, merged), a data-hygiene pass was
  attempted and had to be **reverted** after discovering the files were live
  test fixtures (PR #5, closed unmerged — see §2), and a real, tested,
  leak-free forecasting entrypoint was built from scratch (this branch).
- **The task's actual scored metric is residual normality (Shapiro-Wilk/Francia
  W), not forecast accuracy.** Accuracy is a secondary, judge-facing feature.
- **Honest measured result: 3 of 5 satellite series produce genuine Gaussian
  residuals (H=0).** Mean W = 0.867 across all five, beating the persistence
  baseline (0.855) by a small, real margin.
- A model-selection **data leak was found and fixed** (see §6) — the pre-fix
  number (0.884) was inflated by picking the model that scored best on the
  answer key.
- Two of five series **cannot honestly be made Gaussian** — one is a real
  satellite dynamical divergence, one is small-sample statistical fragility —
  and six different fix attempts were tried and rejected on measured evidence
  before concluding this (§7, §8).
- What remains open: merge to `main`, a UI, retiring the superseded old
  pipeline, two pre-existing broken tests, and enabling branch protection.

---

## 1. What this project is

**OrbitalMind**, Team Xenith, submission SH-DST-03 for Smart Horizon 2026 (a
hackathon). The task, per the organisers' problem statement and `Note.pdf`:

> Given 7 days of GNSS satellite position/clock error history (separately for
> GEO and MEO orbit files), predict the same four error parameters — `x_error`,
> `y_error`, `z_error`, `satclockerror` — at **arbitrary, possibly non-uniform**
> 8th-day timestamps that the organisers will supply. The organisers compute
> `residual = your_prediction − their_hidden_truth` themselves and score its
> **normality**, not just its accuracy.

Scoring, verbatim from `Note.pdf` (extracted and quoted exactly this session):

| Priority | What must be reported | Detail |
|---|---|---|
| **1** | Shapiro-Wilk **W**, **p-value**, **H** (0/1) at α=0.05, averaged equally over the 4 parameters | Higher W is better. **Teams must write their own SW implementation** — not just call a library — and report a confidence interval. |
| **2** | Residual **mean** and **standard deviation** | Tiebreaker if priority-1 scores are equal |
| **3** | **Q-Q plot** | Second tiebreaker, visualizes outliers |

Reference benchmark the scorer must reproduce exactly: **W=0.9810, p=0.5840,
H=0** on the organiser-supplied `data/SW_ReferenceData.xlsx` (45 samples).

**Critical, measured finding:** `scipy.stats.shapiro` (the "obvious" library
call) returns **W=0.9852** on that same reference vector — it does not match
the benchmark. The correct statistic is **Shapiro-Francia** (Blom plotting
positions + Royston-1993 p-value approximation), which reproduces 0.9810/0.5840
to stated precision. This is implemented in `orbitalmind/evaluation/scoring.py`
and pinned by a dedicated test (`test_scoring_contract.py`) specifically so a
"more standard-looking" scipy call can never silently replace it.

---

## 2. Repository state, history, and the cleanup performed this session

### 2.1 The duplicate-folder incident (found and fixed — PR #4, merged)
An unrelated-history merge (`c72effc` "Initial commit: OrbitalMind project" by
a teammate, merged via `3a5fd99`) had nested a **second, stale copy of the
entire project** at `OrbitalMind/OrbitalMind/...` — its own `src/`, `data/`,
`tests/`, `docs/`. Root cause: almost certainly committed from one directory
level too high. Verified byte-for-byte that the nested copy contained nothing
unique (every file was identical to or an older subset of the real copy).
Removed in PR #4 (`chore/remove-duplicate-orbitalmind-folder`), merged clean —
9,490 deletions, 79 files, zero changes outside the duplicate path.

### 2.2 The dataset-cleanup near-miss (PR #5 — closed, NOT merged)
Asked to remove "old dataset" files from the repo. Initial pass deleted:
`data/DATA_GEO_Train.csv`, `DATA_GEO_Test.csv`, `DATA_MEO_Train.csv`,
`DATA_MEO_Train2.csv`, `DATA_MEO_Test.csv`, `DATA_MEO_Test2.csv`, `Note.pdf`,
`SIH_Data_Discription.pdf`, `SW_ReferenceData.xlsx`.

**This was a mistake, caught before merge.** These are not disposable sample
data — they are the actual **PS-08 competition dataset and its official answer
key**, and they are hard-coded, load-bearing fixtures for:
- `src/orbitalmind/ingest.py` and `tests/test_ingest_contract.py` (reference
  `DATA_GEO_Train.csv`, `DATA_MEO_Train.csv`, `DATA_MEO_Train2.csv` by name)
- `tests/test_scoring_contract.py` (reads `SW_ReferenceData.xlsx` to prove the
  scorer reproduces the organisers' benchmark — its own docstring: *"If that
  assertion ever fails, no W number this project reports can be trusted"*)
- `scripts/generate_submission.py` / `scripts/train_and_rank.py` (read the
  `*_Test.csv` files directly)

All the affected tests `pytest.skip()` rather than hard-fail when a fixture is
missing, so deleting them would **not** have shown up as a CI failure (there is
no CI) — it would have silently turned the project's own correctness checks
into no-ops. **PR #5 was closed without merging**, with a comment explaining
why. Nothing on `main` was affected. This is recorded so the same mistake is
not repeated: these 9 files are correct to leave exactly where they are, in
`data/`, tracked in git.

A genuinely separate, correctly-completed piece of that cleanup survived:
- `data/README.md` added, documenting the contract (what's generated vs.
  fetched vs. brought by the user, and that nothing else should be committed).
- `.gitignore` tightened to `data/*` + `!data/README.md`.
- `scripts/generate_synthetic_data.py` + `make synthetic` target added, fixing
  a real, separate, pre-existing bug: `data/synthetic/gnss_synthetic.csv` (the
  smoke-test fixture, unrelated to the PS-08 files above) had gone missing from
  `main` with nothing left to regenerate it, silently breaking the README's own
  quickstart instructions.

### 2.3 GitHub process findings
- `main` has **no branch protection** — the GitHub UI itself flags "Your main
  branch isn't protected." Confirmed: PRs #1, #2, #3 all show **closed, not
  merged** on GitHub, yet their commits are on `main` anyway — meaning
  contributors are pushing directly to `main` and closing PRs as an
  afterthought, not gating merges through review. This is the root cause of
  both incidents in §2.1–2.2: neither the duplicate folder nor several
  undocumented model additions ever went through review.
- **No CI workflow exists** — `.github/` contains only a PR template. The only
  automated check on any PR was CodeRabbit's bot, which skips itself on repos
  with under 10 GitHub stars ("Review skipped: manual review required").
- Three large, mixed-scope commits landed directly on `main` the same day
  (`5801257`, `b408173`, `76d59a1`) adding `HarmonicPredictor`,
  `GaussianProcessPredictor`, `DeepResidualPredictor`, and a batch of unrelated
  test files — one commit alone touched 20 files / 2,432 insertions under a
  message describing only a third of its contents.
- **`GaussianProcessPredictor` directly contradicts a written decision** —
  `docs/DECISIONS.md` Decision 004 explicitly rejected Gaussian Process
  modeling ("scales poorly... GP (scaling issues)") with zero record of that
  rejection being revisited before it was added back. Not corrected this
  session (out of scope); flagged for the team.

---

## 3. Data — full inventory

### 3.1 The current (PS-08) competition dataset — what the deliverable is built on
| File | Rows | Contents |
|---|---|---|
| `data/DATA_GEO_Train.csv` | 142 | GEO satellite, 1 continuous series, no time-jump |
| `data/DATA_GEO_Test.csv` | 69 | GEO day-8 test |
| `data/DATA_MEO_Train.csv` | 90 (stacked) | **Two MEO satellites stacked vertically** — timestamp runs forward, jumps backward once, runs forward again. Splits into 2 series (46 rows fwd / 44 rows fwd) |
| `data/DATA_MEO_Train2.csv` | 244 (stacked) | Same stacking pattern, 2 series (143 / 101 rows) |
| `data/DATA_MEO_Test.csv`, `DATA_MEO_Test2.csv` | small | Matching day-8 test blocks |
| `data/SW_ReferenceData.xlsx` | 45 samples | The organisers' official Shapiro-Wilk benchmark vector — scorer must reproduce W=0.981/p=0.584/H=0 on it |
| `data/Note.pdf` | — | The evaluation-criteria document (quoted in full in §1) |
| `data/SIH_Data_Discription.pdf` | — | Problem-statement / data-description document |

**Non-uniform sampling, confirmed by direct inspection:** GEO training step
sizes include 240s, 840s, 900s, 960s, 1200s, 1260s... — genuinely irregular,
matching the organisers' own statement that both train and test are
non-uniformly sampled.

**The critical, non-obvious parsing bug this project's `ingest.py` fixes:**
each MEO file secretly holds two full-week satellite series stacked on top of
each other, separated by a single backward time-step. A naive loader (this is
exactly what the OLD `run_pipeline.py` still does) labels every row
`SatelliteID="MEO"`, silently fusing two satellites into one series with a
6-day backward jump in the middle. `ingest.py`'s `split_stacked_blocks()`
detects the backward jump and splits correctly, verified against real row
counts (46|44 and 143|101).

### 3.2 The old/superseded data path — real NASA CDDIS pulls
Described in `README.md`/`ARCHITECTURE.md`/`docs/DECISIONS.md`, built earlier
in the project's life, **now superseded by the PS-08 files above** for the
actual deliverable:
- `scripts/fetch_data.py` pulls real multi-GNSS orbit/clock data from **NASA
  CDDIS** (GFZ multi-GNSS rapid combination, `GFZ0MGXRAP` — chosen over the
  IGS rapid combination because IGS is GPS-only/all-MEO and the deck requires
  GEO/GSO satellites too; see Decision 007).
  - Needs Earthdata credentials in `~/.netrc`.
  - Writes 7 days to `data/raw/gnss_real.csv` and a **separate** 8th-day
    holdout file (`data/raw/gnss_holdout.csv`) that training must never read
    (Decision 006).
  - Fails loudly rather than silently substituting synthetic data unless
    `--allow-synthetic` is explicitly passed (a real, deliberately-designed
    safety rail against silently training on the wrong kind of data).
  - Writes a `.provenance.json` recording source products, date spans, counts,
    and `"origin": "real"` vs `"synthetic"`.
- `src/orbitalmind/utils/synthetic_generator.py` generates a small synthetic
  fallback dataset (3 GEO + 5 MEO satellites, 8 days at 15-min intervals) used
  purely for the install smoke test and fast unit tests — never for scoring.
  Not committed to git (regenerated on demand via `make synthetic`, fixed this
  session per §2.2).

**Why the "old data" language surfaced repeated confusion this session:** at
different points "old dataset" was used to mean (a) the CDDIS real-data
pipeline above, (b) the PS-08 files in §3.1, and (c) a user's local
`Data_PS-08` folder never uploaded to this session. All three are distinct.
The PS-08 files in §3.1 are the ones the current, correct deliverable is built
and tested against.

---

## 4. Architecture — OLD pipeline (superseded, still present in the repo)

Entry point `src/orbitalmind/run_pipeline.py`. Full description, verified
against `ARCHITECTURE.md` and the code:

```
CSV → per satellite, per error column (ClockError_ns, EphemerisError_m)
  ├─ preprocess_satellite()          preprocessing/pipeline.py
  │    ├─ remove_outliers_mad()      MAD modified z-score, threshold 3.5
  │    ├─ correct_iod_jumps()        adaptive threshold, robust-MAD-derived
  │    ├─ single_difference()        stationarity via differencing
  │    └─ decompose_signal()         EMD → trend / periodic / noise (PyEMD)
  ├─ combined = trend + periodic     (noise component deliberately discarded)
  ├─ compute_splits()                two-plan windowing (see below)
  └─ per plan (backtest, submission):
       ├─ train 4 base models: LSTM · TCN-LSTM · TFT · Neural ODE
       ├─ forecast → LightGBM meta-learner (stacks the 4 base forecasts)
       ├─ fit a Normalizing Flow on residuals → bias + predictive distribution
       ├─ forecast the target window
       └─ reconstruct to original units, anchored on the last OBSERVED value
```

**The two-plan window design (`splits.py`)** — every window size is derived
from each satellite's actual row count, not a hardcoded constant:
| Plan | train | calibration | target |
|---|---|---|---|
| `backtest` | `[0, n−2h)` | `[n−2h, n−h)` | `[n−h, n)` — real held-out truth, used for honest scoring |
| `submission` | `[0, n−h)` | `[n−h, n)` | `[n, n+h)` — past the end of the file, the actual deliverable |

**The two coordinate frames** (`preprocess_satellite()` returns both):
`observed` (outlier-cleaned only — the measurement frame, used for scoring and
anchoring reconstruction) vs. `original_cleaned` (also IOD-jump-corrected — the
modelling frame, used for differencing/decomposition). Mixing them up
historically cost the project thousands of nanoseconds of error; this is
called out prominently in `ARCHITECTURE.md` as the thing "most likely to be
broken by a well-meaning change."

**Known, documented gaps in the OLD pipeline (pre-existing, not from this
session):**
- Still calls `shapiro_wilk` → `scipy.stats.shapiro`, the **wrong** statistic
  per §1/§4 of this report.
- Still merges the two stacked MEO satellites into one series (the exact bug
  `ingest.py` was built to retire — never wired in).
- All four neural models train on `nn.MSELoss()`, not the Gaussian-likelihood
  loss the original proposal claims.
- `features/` (96-lag + FFT features) and `models/diffusion.py` are fully
  built and tested but **never imported** by `run_pipeline.py` — documented as
  deliberate/orphaned in `ARCHITECTURE.md`, not accidental.
- **This entire pipeline forecasts on a fixed integer 96-step grid** (15-minute
  intervals), not arbitrary timestamps — architecturally incompatible with
  the organisers' actual "predict at arbitrary 8th-day timestamps" requirement
  (Note.pdf 1c/1e). This is the main reason it is superseded, not merely
  outdated.

**Historical performance figures reported in `README.md`** (for the old
CDDIS-data track, not the PS-08 deliverable): sub-nanosecond clock error at
15–30 min horizons beating both persistence and linear baselines; degrading
badly beyond ~2 hours (accumulated autoregressive error); 5.85 ns clock error
at 24 hr vs. a persistence baseline of 1.62 ns (i.e. losing to the trivial
baseline at long horizon). These numbers are **not applicable to the PS-08
deliverable** — different data, different task shape (fixed grid vs. arbitrary
timestamp).

---

## 5. Architecture — NEW pipeline (the actual PS-08 deliverable, current)

### 5.1 Phase 0 — the frozen contracts (`docs/CONTRACTS.md`)
Three small, torch-free (numpy/pandas/scipy only) modules, deliberately frozen
first so five people could build independent "lanes" without blocking on each
other:

1. **`orbitalmind.evaluation.scoring`** — the scorer (§1). Key functions:
   `shapiro_francia(x) -> (W, p, H)`, `score_parameter(name, residuals) ->
   ParameterScore` (adds mean, std, 95% CI), `score_residuals(dict) ->
   ScoreResult` (averages the 4 parameters equally). Pinned by
   `test_scoring_contract.py` (24 tests total across all three contracts).
2. **`orbitalmind.interfaces`** — the `Predictor` protocol:
   `predict(t_query: Sequence[datetime]) -> np.ndarray` of shape `(n, 4)`, in
   `TARGET_COLUMNS` order. Ships a real, deterministic reference stub
   (`PersistencePredictor`, repeats the last observed value) so every
   downstream lane could be built and scored before any real model existed.
   Helper functions `predict_frame()` and `residuals_by_parameter()` bridge to
   the scorer.
3. **`orbitalmind.ingest`** — the loader (§3.1's MEO-stacking fix).
   `load_series(path) -> list[Series]`, `load_dataset(paths) -> list[Series]`
   (flattens multiple files), `split_stacked_blocks()`. Yields exactly **5
   series total**: GEO (1) + MEO_Train (2, blocks 0/1) + MEO_Train2 (2, blocks
   0/1). `Series` is a frozen dataclass exposing `.frame`, `.orbit`,
   `.satellite_id`, `.values() -> (n,4)`.

### 5.2 The three forecaster candidates
All three genuinely implement the `Predictor` protocol; verified by executing
each, not just reading them.

- **`HarmonicPredictor`** (`models/harmonic.py`) — OLS harmonic regression:
  bias + linear drift + 3 harmonics each at 12h and 24h periods (14 parameters
  per target column), fit independently per column via `np.linalg.lstsq`.
  Chosen specifically because the PS-08 series are tiny (42–244 rows) — a
  model with more free parameters than data points would be
  under-determined; this form is always well-conditioned at n≥42 and the
  physics motivate the exact basis.
- **`GaussianProcessPredictor`** (`models/gp_predictor.py`) — sklearn GP with
  a composite kernel: `ExpSineSquared(12h) + ExpSineSquared(24h) +
  Matern(medium-range) + WhiteKernel(noise floor)`. Rationale: harmonics
  capture the dominant periodicity exactly, but the residual has structured
  autocorrelation at the 2–8h scale (atmospheric/orbit-determination
  latency) that a periodic+Matérn kernel can absorb, plus calibrated
  uncertainty. **Measured limitation this session:** GP extrapolates to a
  flat constant far outside the training window (verified: predicting 7 days
  past training end returns identical values for widely-spaced timestamps) —
  fine near day 8, unreliable further out.
- **`DeepResidualPredictor`** (`models/deep_predictor.py`) — a compact
  regularized MLP (`ResidualNeuralHead`, hidden_dim=32, dropout=0.1,
  zero-initialized output layer) trained to predict only the *residual* on
  top of a `HarmonicPredictor` base: `y_pred(t) = y_harmonic(t) +
  f_neural(t;θ)`. The zero-init output layer means if the neural head finds
  no real signal, it contributes ~0 and the prediction smoothly reverts to
  pure harmonic — a deliberate anti-overfitting design for the same
  small-sample constraint.

`scripts/train_and_rank.py` trains and ranks all three (plus the
`PersistencePredictor` baseline) on all 5 series and saves the winner per
series to `models/saved/best_<series_id>.pkl` — this **actually ran
successfully end-to-end this session** (not just read/inspected), producing
real numbers (§6).

### 5.3 The inference entrypoint — `src/orbitalmind/predict.py` (built this session)
The single, real deliverable that ties everything together and is what a
judge or the organisers would actually run:

```
predict.run(train_files, timestamps_path, qq_dir=None)
  │
  ├─ ingest.load_dataset(train_files)           → up to 5 Series
  ├─ _load_query(timestamps_path)                → timestamps, optional truth,
  │                                                  optional satellite_id column
  ├─ per series (routed by satellite_id if the query provides one — fixed
  │              this session, see §9):
  │     ├─ select_model(train)                   → leak-free model choice
  │     ├─ refit winner on the FULL 7-day record
  │     ├─ predict(t_query)                      → (n,4) — THE submitted point
  │     ├─ shaping.calibrate(name, train)         → robust sigma (leak-free)
  │     ├─ interval = point ± 1.96·sigma          → predictive interval columns
  │     └─ if truth present: residual → score_residuals() → W/p/H/mean/std/CI
  │                                               → save_qq_plot() if qq_dir given
  └─ writes outputs/submission.csv, forecast_report.txt/.json, outputs/qq/*.png
```

CLI: `python -m orbitalmind.predict --train <file(s)> --timestamps <file>
[--output ...] [--report ...] [--qq-dir ...]`.

**Why this single path covers both possible evaluation scenarios** (bare
timestamp list, or an entirely new test file): both reduce to exactly "fit on
these 7 days, predict at these timestamps" — verified by feeding it 12
deliberately random, unsorted, non-uniform timestamps spanning up to 7 days
*past* the end of training, well outside day 8; all three model types
returned clean, finite `(n,4)` output.

### 5.4 The shaping/calibration lane — `src/orbitalmind/shaping.py` (built this session)
This module's entire design is an integrity guarantee, not a feature:

- **What it does:** computes a robust (1.4826×MAD) dispersion per parameter,
  fit on a held-out **training-tail** split (never the test/query truth), and
  supplies a symmetric predictive interval (`point ± z·sigma`) plus the
  priority-3 Q-Q diagnostic via `evaluation/gaussian_check.py`.
- **What it deliberately does NOT do:** touch the submitted point forecast at
  all. This is enforced, not just intended:
  - `Calibration` is a frozen dataclass exposing only `sigma`, `source`, and
    `interval()` — no method exists that could alter a point forecast.
  - `tests/test_shaping.py::test_calibration_exposes_no_point_transform`
    asserts the class's public surface is exactly `{sigma, source, interval}`.
  - `tests/test_shaping.py::test_entrypoint_point_forecast_equals_raw_model`
    runs the real entrypoint and asserts the submitted prediction is
    bit-for-bit identical (`atol=0, rtol=0`) to calling the chosen model
    directly.
- **Why not a bias correction (this was built, measured, and removed):** an
  earlier version added a per-parameter additive bias, gated to only apply
  when it reduced validation-tail error. Measured on the real day-8 data, it
  **made the priority-2 residual mean worse on 4 of 5 series** (mean
  |residual mean| across parameters: 0.36 → 0.62 with the bias applied) —
  a bias learned from the calm training week does not transfer through
  day-8's regime change, least of all GEO's divergence. Removed per the
  project's own anti-clutter rule; the decision and the measurement are
  recorded as **Decision 021** in `docs/DECISIONS.md`.

---

## 6. The model-selection leak — found, proven, and fixed

`scripts/train_and_rank.py` originally picked each series' "best" model by
comparing `score_h.W`, `score_g.W`, `score_d.W` — scores computed **on the
test file**, i.e. the answer. At the real evaluation, the answer does not
exist yet, so this number was never achievable in practice.

**Fix:** selection now uses `predict.select_model()`, which splits each
series' **training record** in time (fit on the first ~75%, rank candidates on
the held-out final ~25%) and picks the winner from that alone. The chosen
model is then refit on the *full* training record before it forecasts.

**Measured effect of the fix:**

| | Leaky (pre-fix) | Leak-free (post-fix) |
|---|---|---|
| Mean W across 5 series | 0.884 | **0.867** |
| MEO_Train2-b0 chosen model | GP | Harmonic |
| MEO_Train2-b1 chosen model | Harmonic | GP |

The 0.884 figure was inflated by hindsight, not real generalization skill —
two series' model choices flip once the peek is removed. 0.867 is the number
that would actually be achievable at the real evaluation, where the model
must be chosen before the answer exists.

---

## 7. Measured results, leak-free, on the real day-8 data

| Series | Orbit | n(test) | Chosen model | val_W (leak-free selector) | **test W** | H | Gaussian? |
|---|---|---|---|---|---|---|---|
| GEO_Train-b0 | GEO | 69 | Deep | 0.827 | **0.787** | 1 | ❌ FAIL |
| MEO_Train-b0 | MEO | 5 | Deep | 0.902 | **0.915** | 0 | ✅ PASS |
| MEO_Train-b1 | MEO | 6 | GP | 0.924 | **0.889** | 0 | ✅ PASS |
| MEO_Train2-b0 | MEO | 18 | Harmonic | 0.864 | **0.814** | 1 | ❌ FAIL |
| MEO_Train2-b1 | MEO | 12 | GP | 0.936 | **0.948** | 0 | ✅ PASS |

- **Mean test W = 0.867 · 3/5 series pass normality (H=0)**
- Pooled residuals across all 5 series: W=0.651, H=1 (GEO's failure dominates
  the pooled statistic)
- Persistence-baseline mean W = 0.855 → this build's net honest gain: **+0.012**
  — small, but real and not manufactured
- Per-parameter breakdown for GEO (the worst series), for reference: x_error
  W=0.873, y_error W=0.813, z_error W=0.885, satclockerror W=0.577 (all H=1)

**Full end-to-end proof of capability, run live this session:**
- Fed the entrypoint 12 deliberately random, unsorted, non-uniform timestamps
  up to 7 days past the training window's end → clean `(12,4)` output, all
  finite, from all three model types.
- Loaded a saved model and fed it 6 random out-of-range unsorted timestamps →
  clean `(6,4)` output.
- Confirmed the submitted point forecast is bit-identical before and after the
  shaping lane is applied (the integrity test in §5.4).

---

## 8. Why the two failing series fail — diagnosed with evidence, not guessed

### 8.1 GEO_Train-b0 — a genuine satellite dynamical divergence
Clock-error standard deviation by calendar day, measured directly from the raw
data:

| Day | Sep 1 | Sep 2 | Sep 3 | Sep 4 | Sep 5 | Sep 6 | Sep 7 | **Sep 8 (test)** |
|---|---|---|---|---|---|---|---|---|
| std (m) | 2.28 | 2.05 | 1.57 | 3.32 | 2.76 | 3.43 | 5.46 | **15.90** |
| range (m) | ±4 | ±4 | ±6 | ±5 | ±4 | ±7 | **±23** | **±58** |

The satellite is calm for 5 days, then ramps sharply on day 7 and explodes on
day 8 — consistent with a maneuver, eclipse-season effect, or a clock event.
Any model trained on the calm week predicts ~±2m; reality on day 8 is ±58m, so
the "residual" is mostly the unmodeled divergence itself — a large, structured
signal, and structured signals are inherently non-Gaussian.

**Confirmed the model isn't simply broken:** in-sample RMS on GEO's *own*
training data is 4.95m against a raw training std of 5.02m — i.e. the model
captures essentially zero variance reduction even on data it was fit on,
because the training data has almost no structure to learn relative to its
noise floor; the real signal only appears in the divergence the model never
saw.

**Four independent fix attempts were tried on GEO and all failed** (§ below;
data-driven frequencies, recency weighting, baselines, and transforms) — this
is not for lack of trying.

### 8.2 MEO_Train2-b0 — small-sample single-outlier fragility (a different disease)
Only 18 test points. The `satclockerror` residuals: 17 values cluster tightly
near zero (≤±0.05m) and **one** sits at 0.20m. At n=18, Shapiro-Francia is
highly sensitive — that single point is enough to fail it (that parameter's
W=0.577, H=1 alone). Proof this is outlier-driven rather than a broken model:
progressively dropping the worst 0/1/2/3 points raises pooled W steadily:
0.880 → 0.887 → 0.913 → 0.933. Compare to GEO, where the *entire* signal is
unmodeled — here the model is well-fit and a couple of ordinary-sized points
happen to be enough to fail a test with very little statistical room at n=18.

---

## 9. Approaches tested to fix normality — all measured, all rejected on evidence

| # | Approach | What was measured | Verdict |
|---|---|---|---|
| 1 | Lomb-Scargle data-driven dominant frequencies instead of fixed 12h/24h | GEO real periods are ~2.4–11.9h, not 12/24h. Refit with the "correct" adaptive frequencies: GEO 0.787→0.799 (still H=1); **hurt 3 of 4 MEO series** | Rejected — no net help |
| 2 | Recency weighting (fit on only the most recent training data) | Short fit windows (last 30%, last 15%) caused catastrophic overfitting — clock-residual std exploded to 57,764 and 151,892,440 respectively | Rejected — catastrophic |
| 3 | Persistence / linear-extrapolation baselines on GEO | Both score identically: W=0.771, H=1 | Rejected — no help, GEO's problem isn't model-class-specific |
| 4 | Additive bias correction on the point forecast | W provably unchanged (mathematically affine-invariant); priority-2 mean got **worse** on 4/5 series | Rejected — see §5.4, Decision 021 |
| 5 | Blanket Yeo-Johnson transform on the target before fitting | Helped 2 already-passing series (0.915→0.929, 0.889→0.937), **hurt** a third (0.948→0.918), fixed **zero** failing series | Rejected — net inconsistent |
| 6 | Leak-free per-series selection between transform and no-transform variants | Landed at the same 3/5 pass rate as no transform at all — the selector correctly declined the transform everywhere it didn't help | Rejected — pure clutter for zero gain |

**Conclusion, stated plainly:** no legitimate method — tested across
model-choice, weighting, baseline, and transform axes — improves GEO or
MEO_Train2-b0's normality. 3 of 5 is the honest ceiling on this data. The
project explicitly declined to fake it further (see §10).

---

## 10. The integrity guarantees built into this codebase

1. **You submit predictions, never residuals.** At the real evaluation the
   organisers compute `residual = your_prediction − their_hidden_truth`
   themselves. Any transform applied to a "residual" inside this codebase
   reaches nothing but this project's own internal scorer — it would flatter
   internal numbers while changing the real submission by exactly zero. This
   codebase does not do that anywhere.
2. **Shapiro-Francia's W statistic is provably invariant to location and
   scale** — any purely additive/multiplicative correction to a point forecast
   cannot change it. This is a mathematical fact, not a design choice, and it
   is exploited deliberately: the shaping lane is restricted to exactly such
   corrections so it is *structurally* incapable of gaming W.
3. **Model selection cannot see the answer** — proven in §6, enforced by
   `select_model()`'s time-split design and measured to change the result when
   the leak is removed.
4. **A previous, unrelated instance of exactly this kind of fraud is on
   record** in `docs/DECISIONS.md` — an early iteration manufactured p=0.9999
   by subtracting a ground-truth-derived correction from predictions before
   scoring. `tests/test_normalizing_flow.py` was written specifically to pin
   that this cannot recur (two different prediction vectors must be moved by
   an identical amount; deliberately non-Gaussian residuals must not return a
   high p-value). The shaping lane built this session extends the same
   discipline to the new architecture.
5. Every claim of "fixed" or "measured" in this report and in the commits was
   backed by actually executing code in this session — including installing
   torch/sklearn/scipy in the sandbox and running the real training/inference
   pipeline against the real files, not by reading source and inferring
   behavior.

---

## 11. Test suite — full inventory and status

24 pre-existing "Phase 0 contract" tests + 5 new shaping tests = **29 tests, all
passing**, verified by executing them this session:
- `test_ingest_contract.py` (pins the MEO-stacking split)
- `test_predict_contract.py` (pins the `Predictor` protocol + arbitrary
  timestamp behavior)
- `test_scoring_contract.py` (pins Shapiro-Francia reproducing the org.
  benchmark, and documents why scipy's Shapiro-Wilk would not match)
- `test_shaping.py` (new — the 5 integrity tests from §5.4/§10)

**Broader legacy suite, run this session, with honest caveats:**
- 42 legacy tests (old preprocessing/TFT tests) failed with
  `TypeError: Cannot interpolate with str dtype` — traced to a **pandas 3.0
  behavior change**; this sandbox had numpy 2.4/pandas 3.0 installed while the
  project pins numpy 1.26.4/pandas 2.2.1 in `requirements.txt`. This is a
  sandbox/environment mismatch, not a logic defect, and not something this
  session's code changes caused (no preprocessing/model files were touched).
  **Not yet re-verified on the pinned environment** — listed as an open item.
- **Two pre-existing, genuinely broken tests found on `main`, unrelated to
  this session's work:**
  - `tests/test_score_day8.py` does `import score_day8` — but
    `scripts/score_day8.py` does not exist anywhere in the repository. This
    test cannot even be collected by pytest.
  - `tests/test_causal_preprocess.py` imports `MIN_LIMIT` from
    `orbitalmind.preprocessing.pipeline` — that name does not exist in the
    module. Also fails at collection.
  - Both entered the repo in the same large, unreviewed direct-push commit
    described in §2.3.

---

## 12. Full file manifest — everything touched or newly built this session

| File | Status | Purpose |
|---|---|---|
| `src/orbitalmind/predict.py` | **NEW** | The real inference entrypoint (§5.3) |
| `src/orbitalmind/shaping.py` | **NEW** | The calibration/diagnostics lane (§5.4) |
| `tests/test_shaping.py` | **NEW** | 5 tests incl. the integrity guarantees |
| `scripts/train_and_rank.py` | **FIXED** | Leak-free model selection (§6) |
| `docs/DECISIONS.md` | **APPENDED** | Decision 021 — shaping scope + bias-correction rejection, measured |
| `data/README.md` | Added (earlier pass) | Documents the data/ directory contract |
| `.gitignore` | Tightened (earlier pass) | `data/*` except the README |
| `scripts/generate_synthetic_data.py` | Added (earlier pass) | Fixes the missing-smoke-test-fixture regression |
| `Makefile` | Updated (earlier pass) | Added `make synthetic` target |
| `README.md` | Updated (earlier pass) | Quickstart now generates synthetic data first |

**Deleted, then correctly restored (net: unchanged from before this session):**
`data/DATA_GEO_Train.csv`, `DATA_GEO_Test.csv`, `DATA_MEO_Train.csv`,
`DATA_MEO_Train2.csv`, `DATA_MEO_Test.csv`, `DATA_MEO_Test2.csv`, `Note.pdf`,
`SIH_Data_Discription.pdf`, `SW_ReferenceData.xlsx` — see §2.2.

**Deleted permanently, correctly (PR #4, merged):** the entire duplicate
`OrbitalMind/OrbitalMind/...` tree — see §2.1.

---

## 13. Claims vs. measured reality — the self-comparison table

| Claim / prior assumption | What was actually measured this session |
|---|---|
| "GEO/MEO split is justified by 24h/12h periodicity" (original Decision 002 reasoning) | GEO's actual dominant periods are ~2.4–11.9h; most MEO satellites are 24h-dominant. The periodicity reasoning was already known false per Decision 002's own correction; this session's Lomb-Scargle analysis (§9, row 1) independently reconfirms it. |
| "Residuals are Gaussian by design/engineering" | Never engineered, and 2 of 5 measured series genuinely fail (§7). The 3 that pass, pass honestly. |
| Scorer = "Shapiro-Wilk" implying `scipy.stats.shapiro` | Must be Shapiro-**Francia** specifically — scipy's Shapiro-Wilk gives 0.9852 vs. the required 0.9810 benchmark, a hard mismatch, not a rounding difference. |
| The old 4-model neural ensemble is "the solution" | Superseded for the PS-08 deliverable — it forecasts on a fixed grid, not arbitrary timestamps, and uses the wrong scorer. The actual deliverable is Harmonic/GP/DeepResidual, selected per series. |
| Forecast accuracy is the headline metric | The organisers score **normality of residuals** first; accuracy (mean/std) is priority 2, a tiebreaker only. |
| "Old dataset should be removed for cleanliness" | The PS-08 files are not disposable — they are load-bearing test fixtures and the actual competition data (§2.2). Only the unrelated CDDIS-pipeline concept and stray generated files were fair targets for cleanup. |

---

## 14. What is NOT done — honest, current open items

1. **Not merged to `main`.** All of §5–§10's work lives on
   `feature/inference-entrypoint-and-shaping`. No PR has been opened (per
   explicit instruction this session).
2. **`satellite_id` routing uses this project's internal series-id scheme**
   (e.g. `GEO_Train-b0`). This works cleanly for the organisers' actual
   evaluation shape (separate GEO/MEO files, run once per file — no id column
   needed) and for internal testing. A single *mixed* query file from a judge
   would need its `satellite_id` values to match the training series ids, or
   the entrypoint raises a clear, actionable error (fixed this session — it
   previously crashed with an opaque `pandas.concat` error on a full mismatch;
   now explains exactly which ids were expected).
3. **The old pipeline (`run_pipeline.py` + 4 old neural models + `features/` +
   `diffusion.py`) has not been retired**, only identified as superseded.
   Removing it is a team decision, not made unilaterally.
4. **Two pre-existing broken tests remain unfixed** (§11) — `score_day8.py` is
   still unwritten; `MIN_LIMIT` is still undefined. Both predate this
   session's work.
5. **The 42-legacy-test failure has not been re-verified on the pinned
   dependency versions** (numpy 1.26.4/pandas 2.2.1) — only diagnosed as a
   version-mismatch artifact of this sandbox.
6. **No UI exists yet.** Explicitly the next planned step.
7. **GEO's normality is not fixable**, and is not being further pursued — six
   independent, measured attempts (§9) all failed or made things worse. This
   is reported as a measured physical limitation, not left as a mystery.
8. **`GaussianProcessPredictor`'s contradiction of Decision 004** (§2.3) has
   not been resolved — either the model should be justified with a new
   decision entry, or Decision 004 needs revisiting. Not addressed this
   session; flagged for the team.
9. **Branch protection on `main` is still off.** This is a GitHub
   repository-settings action only a repo admin (Mallhar) can take; it cannot
   be done from this session.

---

## 15. Recommended next steps, in priority order

1. Re-run the full test suite under the **pinned** environment
   (`numpy==1.26.4`, `pandas==2.2.1`) to get an honest legacy-suite baseline.
2. Build the **UI** on top of `predict.py` — a thin shell (upload a training
   file + a timestamp file → get back the submission CSV, the W/p/H report,
   and the Q-Q plot) with zero modeling logic of its own.
3. Team decision: retire or explicitly re-scope the old pipeline
   (`run_pipeline.py` and its four models).
4. Fix or remove the two pre-existing broken tests
   (`test_score_day8.py`, `test_causal_preprocess.py`).
5. Resolve the `GaussianProcessPredictor`/Decision 004 contradiction one way
   or the other, on the record.
6. Merge this branch via a reviewed PR, and turn on branch protection on
   `main` (owner action) so the direct-push pattern that caused §2's incidents
   cannot recur.
