# Checker: Test Review Checklist
# Use this specifically to verify test files are real gates

The maker writes both the code and the tests.
A maker that wants to move fast will write weak tests that always pass.
Your job is to ensure tests would actually catch broken code.

---

## THE TEST REALITY CHECK

For each test function, ask: "If I deleted the function being tested,
would this test still pass?"

If yes — the test is fake. Reject it.
If no — the test is real. Accept it.

---

## SPECIFIC TEST CHECKS PER ITERATION

### Iteration 1 — Synthetic Data
Real tests catch:
- Wrong row count (not just "has rows")
- Wrong column names (exact match not substring)
- NaN values (not just "has values")
- Out of range values (specific bounds not just "is numeric")
Fake tests that must be rejected:
- assert len(df) > 0
- assert df is not None
- assert "Timestamp" in str(df.columns)

### Iteration 2 — Preprocessing
Real tests catch:
- Reconstruction error above 1e-4 (not just "reconstructs")
- IOD correction that makes jumps smaller (not just "runs without error")
- Length exactly original-1 after differencing (not just "shorter")
Fake tests that must be rejected:
- assert result is not None
- assert len(result['trend']) > 0
- assert preprocess_satellite runs

### Iteration 3 — Features
Real tests catch:
- Exactly 29 features (not just "> 0 features")
- GEO has stronger 24hr than 12hr signal (physics check)
- MEO has stronger 12hr than 24hr signal (physics check)
- All values in [0,1] range after normalisation
Fake tests that must be rejected:
- assert X.shape[1] > 10
- assert not np.isnan(X).all()

### Iteration 4 — LSTM
Real tests catch:
- Loss decreases (not just "trains")
- RMSE < 2.0 ns at 1hr (not just "produces predictions")
- Predictions in original scale (not 0-1 range)
- Separate saved files for GEO and MEO
Fake tests that must be rejected:
- assert model is not None
- assert len(predictions) == 96
- assert loss < 999

### Iteration 7 — Normalizing Flow
Real tests catch:
- Shapiro-Wilk p > 0.05 (specific threshold)
- KL divergence < 0.5 (specific threshold)
Fake tests that must be rejected:
- assert p_value > 0
- assert model.log_prob(X) is not None

### Iteration 8 — Pipeline
Real tests catch:
- subprocess returncode == 0 (actual run)
- submission.csv has correct columns (exact names)
- shapiro_wilk_result.txt contains "PASS" (string check)
- All prior tests still pass (regression check)
Fake tests that must be rejected:
- assert os.path.exists("outputs/") 
- assert pipeline_function is not None

---

## IF YOU FIND A FAKE TEST

1. Write in verdict: "Test [name] in [file] is not a real gate — [reason]"
2. Specify what the real test should check instead
3. Mark Section 4 as FAIL
4. Maker must rewrite the test before re-review
