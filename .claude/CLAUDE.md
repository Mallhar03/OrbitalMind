# OrbitalMind — Claude Code Master Instructions
# Team Xenith | SH-DST-03 | Smart Horizon 2026

---

## WHAT THIS PROJECT IS

OrbitalMind is a 6-stage physics-aware ensemble AI pipeline that predicts
GNSS satellite clock and ephemeris errors for Day 8 at 15-minute intervals,
trained on 7 days of satellite error history.

Two error types to predict:
- ClockError_ns: satellite clock bias error in nanoseconds
- EphemerisError_m: satellite orbital position error in metres

Two satellite types that must be handled separately:
- GEO/GSO: geostationary, 24-hour periodicity, slow drift
- MEO: moving orbit, 12-hour periodicity, faster dynamics

---

## READ THIS BEFORE EVERY SESSION

1. Read memory/current_iteration.md — find out which iteration you are on
2. Read memory/what_is_done.md — know what is already built and tested
3. Read memory/what_failed.md — know what was tried and why it failed
4. Read memory/never_do.md — load the hard constraints
5. Read the relevant skill file in skills/ for this iteration
6. Only then touch any code

---

## THE 8 ITERATIONS — BUILD ORDER

Iteration 1: Synthetic data generator
  File: src/orbitalmind/utils/synthetic_generator.py
  Skill: skills/synthetic_data.md
  Test: tests/test_synthetic_data.py
  Done when: generates 672 rows per satellite, GEO and MEO both present,
             correct columns, saves to data/synthetic/gnss_synthetic.csv

Iteration 2: Preprocessing pipeline
  Files: src/orbitalmind/preprocessing/
  Skill: skills/preprocessing.md
  Test: tests/test_preprocessing.py
  Done when: MAD outlier removal works, IOD jumps corrected,
             single difference applied, EWT decomposition produces
             trend + periodic + noise arrays

Iteration 3: Feature engineering
  Files: src/orbitalmind/features/
  Skill: skills/features.md
  Test: tests/test_features.py
  Done when: lag features (t-1,t-4,t-8,t-96) built, FFT spectral
             features show 12hr and 24hr peaks, feature matrix
             shape is correct

Iteration 4: LSTM baseline model
  Files: src/orbitalmind/models/lstm.py
  Skill: skills/lstm.md
  Test: tests/test_lstm.py
  Done when: trains on synthetic data, predicts 96 steps,
             RMSE printed for all 5 horizons, GEO and MEO
             trained separately

Iteration 5: TFT model
  Files: src/orbitalmind/models/tft.py
  Skill: skills/tft.md
  Test: tests/test_tft.py
  Done when: trains and predicts 96 steps, RMSE compared
             against LSTM, TFT must be better at 24hr horizon

Iteration 6: LightGBM meta-learner
  Files: src/orbitalmind/ensemble/lightgbm_meta.py
  Skill: skills/meta_learner.md
  Test: tests/test_meta_learner.py
  Done when: fuses LSTM + TFT outputs, RMSE improves over
             best single model

Iteration 7: Normalizing Flow post-processor
  Files: src/orbitalmind/models/normalizing_flow.py
  Skill: skills/normalizing_flow.md
  Test: tests/test_normalizing_flow.py
  Done when: Shapiro-Wilk p > 0.05 on residuals,
             KL divergence computed and printed

Iteration 8: Full pipeline integration
  Files: src/orbitalmind/run_pipeline.py
  Skill: skills/pipeline.md
  Test: tests/test_pipeline.py
  Done when: single command runs all stages end to end,
             outputs submission CSV + evaluation report,
             no errors, reproducible with fixed random seeds

---

## VERIFY CONDITIONS — v1 PASSES WHEN

1. pytest tests/ runs with zero failures
2. RMSE at 1hr horizon < 2.0 ns on synthetic data
3. Shapiro-Wilk p > 0.05 on final residuals
4. python src/orbitalmind/run_pipeline.py runs without error

---

## PROJECT FOLDER STRUCTURE

OrbitalMind/
├── .claude/
│   ├── CLAUDE.md               ← this file
│   └── hooks/
├── memory/
├── skills/
├── data/
│   ├── synthetic/
│   └── raw/
├── src/
│   └── orbitalmind/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       ├── ensemble/
│       ├── evaluation/
│       ├── utils/
│       └── run_pipeline.py
├── tests/
├── notebooks/
├── requirements.txt
└── README.md

---

## TECH STACK

Language: Python 3.10+
Deep learning: PyTorch 2.x
Sequence models: pytorch-forecasting (TFT)
Neural ODE: torchdiffeq
Normalizing Flow: normflows
Meta-learner: lightgbm
Signal decomposition: EMD-signal (PyEMD)
Statistical evaluation: scipy.stats
Data: pandas, numpy
Visualisation: matplotlib, seaborn
Testing: pytest
Compute: local Linux machine + Google Colab T4 (GPU-heavy steps)

---

## CODING STANDARDS

- Every function must have a docstring explaining args and return values
- Every module must have a corresponding test file
- All random operations use seed=42
- No hardcoded paths — use config or relative paths
- Never print inside functions — use return values
- Run black formatter before marking any iteration complete

---

## SUB-AGENT: THE CHECKER

You (the maker) cannot approve your own work.
After pytest passes for an iteration, you do NOT mark it complete.
You stop and request a checker review.

How the checker works:
- Checker is a SEPARATE Claude Code session
- Checker reads .claude/checker/CHECKER.md
- Checker reviews your code against review_code.md and review_tests.md
- Checker writes verdict to memory/checker_verdict.md
- Only a VERDICT: PASS in that file means iteration is complete

Your job ends when pytest passes.
The checker's job begins there.
Do not conflate the two.

---

## GRACEFUL SESSION HANDLING

### At the start of EVERY session — no exceptions
Run this before touching any file:
    python scripts/resume.py

Read the RESUME INSTRUCTION it prints.
Follow it exactly. Do not assume you know where you left off.

### After writing every single function
Run this immediately after the function works:
    python scripts/write_checkpoint.py \
      --iteration N \
      --function "function_name" \
      --status complete \
      --note "what it does and what was verified"

Do not write the next function before running this.
If you stop mid-function, run it with --status in_progress.

### If you are running low on context
Signs you are near token limit:
- Responses getting shorter
- Starting to lose track of earlier decisions
- Repeating questions you already answered

When you notice this, before stopping:
1. Write checkpoint for current function with --status in_progress
2. Update memory/current_iteration.md with exact next action
3. Run: git add . && git commit -m "WIP: Iteration N — [function name] in progress"
4. Write one line to memory/what_is_done.md: "SESSION ENDED mid-[function]"
5. Stop cleanly

### When you open a new session after a stop
1. Run python scripts/resume.py
2. Read the full output
3. Do NOT start fresh — continue from checkpoint
4. Do NOT rewrite functions already marked [x] in checkpoint
