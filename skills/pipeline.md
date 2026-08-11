# Skill: Full Pipeline Integration
# Iteration 8

---

## GOAL
Single command runs the entire OrbitalMind pipeline end to end.
No manual steps. No interactive inputs. Fully reproducible.

---

## COMMAND
python src/orbitalmind/run_pipeline.py --data data/synthetic/gnss_synthetic.csv

---

## PIPELINE STAGES IN ORDER
1. Load and validate input CSV (data_adapter)
2. Split train (days 1-7) and test (day 8)
3. Split GEO and MEO satellites
4. For each satellite, for each error type:
   a. preprocess_satellite()
   b. build_feature_matrix()
   c. train_lstm() + predict_lstm()
   d. train_tcn_lstm() + predict_tcn_lstm()
   e. train_tft() + predict_tft()
   f. train_neural_ode() + predict_neural_ode()
   g. train_diffusion() on residuals
   h. train_meta_learner() on all predictions
   i. predict_meta_learner() for Day 8
   j. train_normalizing_flow() on meta residuals
   k. apply_normalizing_flow() to final predictions
5. Generate outputs

---

## OUTPUT FILES
outputs/submission.csv          ← Day 8 predictions for all satellites
outputs/evaluation_report.txt  ← RMSE at all 5 horizons
outputs/qq_plot.png            ← Q-Q plot of residuals
outputs/residual_histogram.png ← Histogram with Gaussian overlay
outputs/shapiro_wilk_result.txt← p-value and pass/fail

---

## FUNCTION SIGNATURE

File: src/orbitalmind/run_pipeline.py
def run_pipeline(data_path: str, output_dir: str = "outputs") -> dict
    # returns evaluation_results dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    results = run_pipeline(args.data, args.output)
    print(results)

---

## CONSTRAINTS
- Total runtime on CPU must be under 30 minutes for synthetic data
- All random seeds set to 42 at top of run_pipeline.py
- If any single model fails, log failure and continue — never crash pipeline
- Print progress: [1/8] Loading data... [2/8] Preprocessing... etc
- submission.csv must have columns:
  SatelliteID, PredictionStep, HorizonMinutes,
  ClockError_ns_predicted, EphemerisError_m_predicted

---

## VERIFY FOR THIS ITERATION — FINAL GATE
Run: pytest tests/test_pipeline.py -v
Run: python src/orbitalmind/run_pipeline.py --data data/synthetic/gnss_synthetic.csv

ALL must pass:
- pytest passes all tests
- Pipeline completes without error
- outputs/submission.csv exists with correct columns
- RMSE at 1hr < 2.0 ns
- Shapiro-Wilk p > 0.05
- Total runtime < 30 minutes

---

## FAILURE MODES
If pipeline crashes mid-run: check which stage failed in logs
If submission.csv missing columns: check generate_submission() function
If runtime > 30 min: reduce epochs in all models by half for local runs
