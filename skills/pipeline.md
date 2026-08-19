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
1. Load input CSV, validate required columns
2. Per satellite: read OrbitType from the COLUMN (never from the ID prefix)
3. For each satellite, for each error type:
   a. preprocess_satellite()   -> observed + original_cleaned + IMFs
   b. compute_splits(len(combined))  -> backtest and submission plans
   c. for each plan:
        train_lstm / train_tcn_lstm / train_tft / train_neural_ode
          on EXACTLY plan.train
        forecast plan.cal
        train_meta_learner()        on the cal_meta half
        train_normalizing_flow()    on the cal_flow half (out-of-sample)
        forecast plan.target from plan.input
        apply_normalizing_flow()    scalar bias correction
        predictive_interval()       sigma and 95% bounds
        reconstruct anchored on observed[t0]
   d. backtest -> RMSE in ns/m vs persistence AND linear baselines
      submission -> the rows written to submission.csv
4. Generate outputs

NOT YET WIRED IN, though the proposal credits both:
   build_feature_matrix()  -- features/ is orphaned
   train_diffusion()       -- diffusion.py is orphaned; the fourth ensemble
                              member is currently a plain LSTM

---

## OUTPUT FILES
outputs/submission.csv          ← Day 8 predictions, plus sigma and 95% bounds
                                   per error type (11 columns)
outputs/evaluation_report.txt  ← RMSE at all 5 horizons in ns/m, against BOTH
                                   the persistence and linear baselines
outputs/qq_plot.png            ← Q-Q plot of residuals
outputs/residual_histogram.png ← Histogram with Gaussian overlay
outputs/shapiro_wilk_result.txt← p-value and pass/fail

---

## FUNCTION SIGNATURE

File: src/orbitalmind/run_pipeline.py
def run_pipeline(data_path: str, output_dir: str = "outputs",
                 backtest: bool = True, max_satellites: int = 0) -> dict
    # returns rmse_ns, baseline_rmse_ns, shapiro_wilk_p,
    #         shapiro_wilk_result, fallbacks

# CLI flags: --data --output --no-backtest --max-satellites

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
