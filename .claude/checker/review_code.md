# Checker: Code Review Checklist
# Use this for every iteration review

Work through every item in order.
Mark each as PASS or FAIL with one line of evidence.
A single FAIL means the whole iteration is rejected.

---

## SECTION 1: Physics Constraints
# These are the most critical. Wrong physics = wrong predictions.

[ ] 1.1 sequence_length == 96 in every model training function
    Evidence: grep -n "sequence_length" src/orbitalmind/models/*.py
    Fail condition: any value other than 96

[ ] 1.2 GEO and MEO trained as separate model instances
    Evidence: check training calls — two separate train() calls, not one with orbit_type as feature
    Fail condition: single model trained on combined GEO+MEO data

[ ] 1.3 IOD correction subtracts offset from jump point forward
    Evidence: read correct_iod_jumps() — check slice is [pos:] not just [pos]
    Fail condition: offset not propagated forward

[ ] 1.4 single_difference stores first_value and returns it
    Evidence: check return statement of single_difference()
    Fail condition: first_value not returned or not stored in output dict

[ ] 1.5 np.random.seed(42) present before synthetic data generation
    Evidence: grep -n "seed" src/orbitalmind/utils/synthetic_generator.py
    Fail condition: seed not set or set after first random call

[ ] 1.6 torch.manual_seed(42) present at top of every train function
    Evidence: grep -n "manual_seed" src/orbitalmind/models/*.py
    Fail condition: any train function missing seed

[ ] 1.7 Training never uses day 8 data
    Evidence: check train/val split — split must be at index 672 (7 days × 96)
    Fail condition: any data beyond index 671 used in training

---

## SECTION 2: Output Contracts
# If output format is wrong, next iteration breaks silently.

[ ] 2.1 preprocess_satellite() returns dict with exactly these 8 keys:
    sat_id, error_col, trend, periodic, noise,
    first_value, original_cleaned, timestamps
    Evidence: read return statement of preprocess_satellite()
    Fail condition: any key missing or extra keys present

[ ] 2.2 build_feature_matrix() returns exactly 29 features
    Evidence: run shape check or count features in feature_names list
    Fail condition: X.shape[1] != 29

[ ] 2.3 predict_*() functions return np.ndarray of shape (96,)
    Evidence: check return type annotation and shape of output
    Fail condition: wrong shape or wrong type

[ ] 2.4 Predictions are in original scale (not normalised)
    Evidence: check that inverse_transform is applied before return
    Fail condition: values are between 0 and 1 (still normalised)

[ ] 2.5 submission.csv has exactly these columns:
    SatelliteID, PredictionStep, HorizonMinutes,
    ClockError_ns_predicted, EphemerisError_m_predicted
    Evidence: read generate_submission() column definitions
    Fail condition: any column missing or renamed

---

## SECTION 3: Rule Compliance
# Check memory/never_do.md against code

[ ] 3.1 No sequence_length below 96
    grep -rn "sequence_length" src/ — verify all values
    
[ ] 3.2 No GEO/MEO mixing in model training
    Check every train() call for orbit_type handling

[ ] 3.3 No hardcoded absolute paths
    grep -rn "\/home\/" src/ — must return empty
    grep -rn "\/mnt\/" src/ — must return empty

[ ] 3.4 No interactive input() calls
    grep -rn "input(" src/ — must return empty

[ ] 3.5 No missing docstrings
    Every function must have """ docstring """
    grep -rn "def " src/ — check each one has a docstring on next line

[ ] 3.6 No random calls without seed
    grep -rn "np.random\." src/ — each must be preceded by seed
    grep -rn "torch.rand" src/ — each must be in seeded context

---

## SECTION 4: Test Quality
# Bad tests are worse than no tests — they give false confidence

[ ] 4.1 Every assert has a meaningful threshold
    Check: assert rmse < 2.0 ✓ (specific threshold)
    Reject: assert rmse > 0 ✗ (always true)
    Reject: assert result is not None ✗ (trivially true)

[ ] 4.2 Test fixtures use real pipeline outputs not mocked data
    Check: fixtures call actual generate_synthetic_gnss_data()
    Reject: fixtures use np.random.randn() as fake input

[ ] 4.3 RMSE thresholds in tests match skill file specifications exactly
    Skill file says 1hr RMSE < 2.0 ns
    Test must assert rmse['1hr'] < 2.0
    Fail condition: test uses different threshold than skill file

[ ] 4.4 Shapiro-Wilk test checks p > 0.05 not just p > 0
    Fail condition: wrong threshold or missing test

[ ] 4.5 Each test file has at least 8 test functions
    Fail condition: fewer than 8 tests means coverage is shallow

---

## SECTION 5: Code Quality

[ ] 5.1 black formatting applied
    Run: black --check src/ — must return "All done!"
    Fail condition: any formatting errors

[ ] 5.2 No print statements inside functions
    grep -rn "print(" src/ — all prints must be at module level or in run_pipeline.py only

[ ] 5.3 All imports are at top of file
    Fail condition: imports inside functions

[ ] 5.4 requirements.txt includes every imported library
    Check every import against requirements.txt
    Fail condition: any import not in requirements.txt

---

## SECTION 6: Integration Check
# Does this module connect correctly to the next iteration?

[ ] 6.1 Output of this iteration matches expected input of next iteration
    Check the skill file for the NEXT iteration
    Verify output format matches what next iteration's INPUT section specifies

[ ] 6.2 No circular imports
    Module in iteration N must not import from iteration N+1
    preprocessing cannot import from models
    features cannot import from ensemble

---

## VERDICT TEMPLATE

Copy this to memory/checker_verdict.md after review:

---
Iteration: [N]
Review date: [date]
Reviewer: Checker Agent

VERDICT: [PASS / FAIL]

Section 1 Physics: [PASS/FAIL] — [evidence or failure reason]
Section 2 Contracts: [PASS/FAIL] — [evidence or failure reason]
Section 3 Rules: [PASS/FAIL] — [evidence or failure reason]
Section 4 Tests: [PASS/FAIL] — [evidence or failure reason]
Section 5 Quality: [PASS/FAIL] — [evidence or failure reason]
Section 6 Integration: [PASS/FAIL] — [evidence or failure reason]

Failures requiring fix before re-review:
1. [specific file, line number, exact problem]
2. [specific file, line number, exact problem]

Re-review requested: [Yes/No]
---
