# Phase 0 — the frozen contracts, and the lanes they unlock

This is the specification the team codes against. Once these three interfaces
are pinned by tests (they now are), the pipeline stops being a chain where each
person waits on the person before them, and becomes five lanes that run at the
same time without touching each other's code.

Nothing here is a model. It is the small, indivisible core — the interfaces and
file formats — that has to exist *before* the work can be divided. It is built
and tested with numpy / pandas / scipy only (no torch), so any machine can run
and verify it.

---

## Why this exists (the division problem, settled)

Three ways to split the work were on the table:

- **By model** — mutually exclusive, but not collectively exhaustive. Four
  models plus "integrate at the end" never produces the scorer, the stacked
  parser, arbitrary-timestamp prediction, the residual layer, or the report.
- **By dataset** — perfectly MECE, and wrong: five people rebuild the same
  machinery on 331 total rows.
- **By stage** — collectively exhaustive, but a dependency chain: the forecaster
  needs the data layer; the shaping layer needs the forecaster *and* the scorer.

A pipeline is a chain **only until its interfaces are frozen**. Freeze two
contracts and the chain becomes independent modules:

```
score(residuals_by_parameter)          -> W, p, H  (per parameter, then averaged)
predict(t_query)                       -> (n, 4) array of x, y, z, clock in metres
```

Everyone downstream then codes to a specification instead of to each other's
half-finished code.

---

## Contract 1 — Scoring  (`orbitalmind.evaluation.scoring`)

The single admissible scorer. Every W number that steers a decision comes from
here. It reproduces the organisers' benchmark (`data/SW_ReferenceData.xlsx`,
Note.pdf): **W 0.981, p 0.584, H 0** at α = 0.05.

**Key fact, measured:** the metric is **Shapiro-Francia** (Blom plotting
positions + Royston-1993 p-value), *not* `scipy.stats.shapiro`, which returns
0.9852 on the same reference vector and therefore does not score this task.

```python
shapiro_francia(x)                    -> (W, p, H)          # the pinned primitive
score_parameter(name, residuals)      -> ParameterScore     # W,p,H,mean,std,CI
score_residuals({param: residuals})   -> ScoreResult        # averaged over params
```

- Four parameters — `x_error, y_error, z_error, satclockerror` — equal weight.
- `ParameterScore` carries priority-1 (W, p, H), priority-2 (mean, std) and the
  95 % CI on the mean residual. Q-Q plots (priority 3) live in
  `evaluation.gaussian_check`.
- Residual convention: `predicted - truth` (W and p are sign-invariant).

Pinned by `tests/test_scoring_contract.py`.

## Contract 2 — Prediction  (`orbitalmind.interfaces`)

```python
class Predictor(Protocol):
    def predict(self, t_query) -> np.ndarray   # shape (len(t_query), 4), metres
```

- Columns are `TARGET_COLUMNS` order — the same order truth uses, so a
  prediction and its truth line up with no renaming.
- `t_query` is arbitrary 8th-day timestamps (may be non-uniform, unsorted) — the
  exact evaluation form from Note.pdf.
- `PersistencePredictor` is the reference **stub**: a real, deterministic
  predictor (repeats the last observed value). Lanes build and score against it
  before any model exists. `residuals_by_parameter(pred, truth)` and
  `predict_frame(pred, t_query)` bridge to the scorer and the report.

Pinned by `tests/test_predict_contract.py`.

## Contract 3 — Ingest  (`orbitalmind.ingest`)

```python
load_series(path)         -> list[Series]     # GEO -> 1, each MEO file -> 2
load_dataset([paths...])  -> list[Series]     # the three train files -> 5 series
split_stacked_blocks(df)  -> list[df]
```

- **Each MEO file holds two satellites stacked vertically**, both spanning the
  same week, separated by a single backward time step. The split happens at that
  step. GEO has no step and stays one series. Result: **five series** —
  GEO, MEO-1 b0/b1, MEO-2 b0/b1.
- `Series` exposes `.frame` (Timestamp + 4 metre columns, strictly increasing
  time), `.orbit`, `.satellite_id`, and `.values()` → `(n, 4)`.
- Header whitespace is normalised (`DATA_MEO_Train.csv` ships `y_error  (m)`).

Pinned by `tests/test_ingest_contract.py`.

---

## Two live bugs these contracts retire

Both are in `src/orbitalmind/run_pipeline.py` and are why the contracts are
worth freezing before dividing work:

1. **Wrong scorer.** `run_pipeline.py` calls `shapiro_wilk` →
   `scipy.stats.shapiro` (line ~322 in `models/normalizing_flow.py`, used at
   ~631 and ~660). It computes 0.9852 where the evaluator computes 0.981. The
   **delivery lane** replaces those call sites with `evaluation.scoring`.
2. **Merged MEO series.** `run_pipeline.py:520-521` labels every row of a MEO
   file with a single `SatelliteID = "MEO"`, silently fusing the two stacked
   satellites into one series with a six-day backward step in the middle. The
   **data lane** routes ingest through `orbitalmind.ingest`.

---

## The five independent lanes

Each depends only on the contracts above, never on another lane's progress.

| Lane | Owns | Done when |
|------|------|-----------|
| **Data** | route the pipeline through `ingest`; retire the MEO-merge bug; expose any queried timestamp | all six files load into `Series`; a test catches the backward jump *(the ingest contract already provides and tests this — this lane wires it into `run_pipeline`)* |
| **Forecaster** | models implementing `Predictor` (harmonic/trend, GP, Neural ODE, …) | ≥3 predictors implement `predict()` and are ranked by leave-one-series-out W against the stub |
| **Transfer (GPU)** | pre-train on the ~64k-row CDDIS archive, fine-tune on the shipped rows, behind `Predictor` | a fine-tuned predictor beats the best forecaster on held-out W, *or* the lane is killed with a documented negative result |
| **Shaping** | everything between a point forecast and residuals: bias, dispersion, diagnostics | dispersion parameter chosen from training days only, both variants measured on the harness |
| **Delivery** | the report the organisers asked for: W/p/H + mean/std + CI + Q-Q, at arbitrary timestamps; retire the wrong-scorer bug | full report generates end-to-end from the stub prediction |

**Two rules on top of MECE**, both measured, not stylistic:

- *Anti-orphan* — nothing ships unintegrated.
- *Anti-clutter* — nothing stays in the pipeline unless it measurably improves
  the harness score. Four models all wired in and all making no difference is a
  failure MECE alone does not catch.

And the verification principle the council converged on:

> **MECE over code artefacts. Deliberate redundancy over any number that changes
> the plan.** The one figure nobody re-derives is the one that quietly steers
> everyone wrong.

---

## Run the contract tests

```bash
python -m pytest tests/test_scoring_contract.py tests/test_ingest_contract.py tests/test_predict_contract.py -q
```

No torch required. 24 tests; the benchmark-reproduction test is the load-bearing
one.
