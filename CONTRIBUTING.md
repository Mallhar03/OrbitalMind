# Contributing to OrbitalMind

We have days, not months, and one audit already found seven defects that a
fully green test suite did not catch. These rules exist so review time goes to
the things that actually broke this project before.

**Every change goes through a pull request. Nobody pushes to `main`.**

---

## The merge bar

A PR merges when **all** of these hold. If any fails, it does not merge — no
exceptions for "it's a small change" or "we're short on time". Short on time is
exactly when this matters.

### 1. Tests pass, and the new behaviour has a test

```bash
pytest tests/ --ignore=tests/test_pipeline.py -q
```

A bug fix must come with a test that **fails before the fix and passes after**.
State that in the PR. "Added tests" is not enough — say which test would have
caught the bug.

### 2. No accuracy claim without a measurement

Any PR claiming to improve results must include, in the description:

- RMSE **in ns or metres**, never in differenced or filtered space
- measured on the **held-out backtest**, not on training or calibration data
- compared against **linear extrapolation**, not just persistence
- the before number and the after number

"Seems better" is not a result. Neither is a number with no baseline.

### 3. Nothing fails silently

A PR is rejected if it adds any of these:

- a bare `except:` or an `except Exception:` that swallows and continues
- a fallback to placeholder values (zeros, means, synthetic data) without a
  loud warning **and** a record in the output
- a default that hides a missing input

We shipped a pipeline that wrote `np.zeros(96)` into the submission on failure
and produced a complete-looking CSV. And a data fetcher that silently
substituted synthetic data. Both passed review. Both were nearly catastrophic.

### 4. No hardcoded assumptions about data shape

Rejected on sight:

- literal row counts or window indices (`480`, `576`, `672`, `[:96]`)
- absolute physical thresholds not derived from the data (`2.0` ns for jump
  detection, `5.0` m clipping)
- satellite identity inferred from an ID string when a column exists
- paths relative to the working directory instead of `orbitalmind.paths`

Windows come from `orbitalmind.splits`. Thresholds come from the data's own
robust statistics. **These four categories caused four of the seven defects.**

### 5. No metric may be computed from the thing it scores

The single most important rule. A post-processor may **not** use the ground
truth of the window it is later evaluated against.

We shipped a normalizing flow that derived its correction from validation truth,
making Shapiro-Wilk return p = 0.9999 for every possible input — including
residuals that were half constant, half exponential. It measured nothing for
eight iterations.

If a PR touches evaluation or post-processing, the description must state what
data the correction was fitted on and why that is out-of-sample.

**Shapiro-Wilk is reported, never engineered.** A PR that makes it pass by
changing the post-processor rather than the model is rejected.

### 6. Docstrings, and docs updated

Every function: Args and Returns. If behaviour changes, `ARCHITECTURE.md`,
`README.md` and the relevant `skills/*.md` change in the same PR. We had six
docs describing code that no longer existed.

### 7. Scope

One concern per PR. A PR that fixes a bug *and* refactors *and* adds a feature
gets sent back to be split. Reviews get sloppy when they get long.

---

## Review checklist for the reviewer

Do not approve until you can answer each of these:

- [ ] What test would have caught the bug this fixes?
- [ ] If it claims an improvement — measured how, against which baseline, in what units?
- [ ] Can this fail without anyone noticing?
- [ ] Does it assume a row count, sampling interval, ID format, or physical constant?
- [ ] Does any evaluation code touch the truth it is scored against?
- [ ] Would a stranger reading only the docs be misled?

**If you cannot answer, ask. Do not approve.** Approving something you do not
understand is how all seven defects survived eight iterations of review.

---

## Reading order for a change

1. `ARCHITECTURE.md` sections 2 and 3 — the two window plans, the two
   coordinate frames. Most likely to be broken by a well-meaning change.
2. `memory/what_failed.md` — the seven defects with root causes. Read before
   trusting any older claim in this repo.
3. `memory/never_do.md` — hard constraints.
4. `memory/decisions.md` — settled decisions. Do not relitigate without new
   evidence.
