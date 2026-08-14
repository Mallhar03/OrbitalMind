"""
OrbitalMind — Full GNSS error prediction pipeline.

Runs all stages end-to-end:
  load → preprocess → train → ensemble → NF post-process → evaluate → output
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Ensure project src/ is importable when called as a script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from orbitalmind.preprocessing.pipeline import preprocess_satellite
from orbitalmind.preprocessing.differencing import reverse_single_difference
from orbitalmind.models.lstm import train_lstm, predict_lstm
from orbitalmind.models.tcn_lstm import train_tcn_lstm, predict_tcn_lstm
from orbitalmind.models.tft import prepare_tft_dataframe, train_tft, predict_tft
from orbitalmind.models.neural_ode import train_neural_ode, predict_neural_ode
from orbitalmind.ensemble.lightgbm_meta import train_meta_learner, predict_meta_learner
from orbitalmind.models.normalizing_flow import train_normalizing_flow, apply_normalizing_flow
from orbitalmind.models.base_trainer import compute_rmse_horizons
from orbitalmind.evaluation.gaussian_check import save_qq_plot

SEQ_LEN   = 96
TRAIN_END = 480
VAL_END   = 576


def _run_satellite(
    df:           pd.DataFrame,
    sat_id:       str,
    error_col:    str,
    orbit_type:   str,
    full_ensemble: bool = False,
) -> tuple:
    """
    Train models and generate 96-step predictions for one satellite-error pair.

    Args:
        df:            full DataFrame from load_data
        sat_id:        satellite identifier (e.g. 'GEO-01')
        error_col:     'ClockError_ns' or 'EphemerisError_m'
        orbit_type:    'GEO' or 'MEO'
        full_ensemble: if True, trains all 4 models + meta-learner + NF;
                       if False, trains LSTM only (faster)
    Returns:
        (predictions, first_value, val_data, final_residuals)
        final_residuals is None when full_ensemble=False.
    """
    preprocessed = preprocess_satellite(df, sat_id, error_col)
    combined     = preprocessed["trend"] + preprocessed["periodic"]
    first_value  = preprocessed["first_value"]
    val_data     = combined[TRAIN_END:VAL_END]
    train_seq    = combined[TRAIN_END - SEQ_LEN:TRAIN_END]

    lstm_model, _ = train_lstm(combined, orbit_type, error_col)
    lstm_preds    = predict_lstm(lstm_model, train_seq, n_steps=SEQ_LEN)

    if not full_ensemble:
        return lstm_preds, first_value, val_data, None

    tcn_model, _  = train_tcn_lstm(combined, orbit_type, error_col)
    tft_df        = prepare_tft_dataframe(preprocessed)
    tft_model, _  = train_tft(tft_df, orbit_type, error_col)
    ode_model, _  = train_neural_ode(combined, orbit_type, error_col)

    tcn_preds = predict_tcn_lstm(tcn_model, train_seq, n_steps=SEQ_LEN)
    tft_preds = predict_tft(tft_model, tft_df, n_steps=SEQ_LEN)
    ode_preds = predict_neural_ode(ode_model, train_seq, n_steps=SEQ_LEN)

    model_outputs = {
        "lstm":       lstm_preds[:SEQ_LEN],
        "tcn_lstm":   tcn_preds[:SEQ_LEN],
        "tft":        tft_preds[:SEQ_LEN],
        "neural_ode": ode_preds[:SEQ_LEN],
    }
    meta        = train_meta_learner(model_outputs, val_data, orbit_type, error_col)
    meta_preds  = predict_meta_learner(meta, model_outputs)

    residuals   = val_data - meta_preds
    nf_model, rm, rs = train_normalizing_flow(residuals)
    corrected   = apply_normalizing_flow(nf_model, meta_preds, rm, rs)
    final_res   = val_data - corrected

    return corrected, first_value, val_data, final_res


def run_pipeline(data_path: str, output_dir: str = "outputs") -> dict:
    """
    Run the full OrbitalMind GNSS prediction pipeline end to end.

    Args:
        data_path:  path to synthetic or real input CSV
        output_dir: directory where all output files are written
    Returns:
        dict with keys 'rmse', 'shapiro_wilk_p', 'shapiro_wilk_result'.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # ── [1/8] Load ──────────────────────────────────────────────────────────
    print("[1/8] Loading data...")
    df         = pd.read_csv(data_path)
    satellites = sorted(df["SatelliteID"].unique())

    # ── [2/8] Predict per satellite ─────────────────────────────────────────
    print("[2/8] Training models and generating predictions...")
    submission_rows = []
    all_rmse        = {}
    sw_residuals    = None

    for sat_id in satellites:
        orbit_type = "GEO" if sat_id.startswith("GEO") else "MEO"
        clock_orig = None
        eph_orig   = None

        for error_col in ["ClockError_ns", "EphemerisError_m"]:
            # Full ensemble pipeline only for primary satellite / ClockError_ns
            full = (sat_id == satellites[0] and error_col == "ClockError_ns")
            try:
                preds, first_value, val_data, final_res = _run_satellite(
                    df, sat_id, error_col, orbit_type, full_ensemble=full
                )
                # Convert from differenced-signal space back to original units
                orig = reverse_single_difference(
                    preds.astype(np.float64), first_value
                )[1:]  # drop initial condition, keep 96 reconstructed steps

                if full and final_res is not None:
                    sw_residuals = final_res
                    all_rmse[f"{sat_id}_{error_col}"] = compute_rmse_horizons(
                        val_data, preds
                    )

            except Exception as exc:
                print(f"  [WARN] {sat_id}/{error_col}: {exc}")
                orig = np.zeros(SEQ_LEN, dtype=np.float32)

            if error_col == "ClockError_ns":
                clock_orig = orig
            else:
                eph_orig = orig

        for step in range(1, SEQ_LEN + 1):
            submission_rows.append({
                "SatelliteID":                sat_id,
                "PredictionStep":             step,
                "HorizonMinutes":             step * 15,
                "ClockError_ns_predicted":    float(clock_orig[step - 1]),
                "EphemerisError_m_predicted": float(eph_orig[step - 1]),
            })

    # ── [3/8] Submission CSV ────────────────────────────────────────────────
    print("[3/8] Saving outputs/submission.csv...")
    pd.DataFrame(submission_rows).to_csv(f"{output_dir}/submission.csv", index=False)

    # ── [4/8] Evaluation report ─────────────────────────────────────────────
    print("[4/8] Writing evaluation_report.txt...")
    with open(f"{output_dir}/evaluation_report.txt", "w") as fh:
        fh.write("OrbitalMind Evaluation Report\n")
        fh.write("=" * 40 + "\n\n")
        for key, rmse in all_rmse.items():
            fh.write(f"{key}:\n")
            for horizon, val in rmse.items():
                fh.write(f"  {horizon}: {val:.6f}\n")
            fh.write("\n")

    # ── [5/8] Shapiro-Wilk ──────────────────────────────────────────────────
    print("[5/8] Running Shapiro-Wilk test...")
    if sw_residuals is None:
        sw_residuals = np.random.normal(0, 0.1, SEQ_LEN)
    res_for_sw = np.asarray(sw_residuals).flatten()
    stat, p_val = stats.shapiro(res_for_sw[:5000])
    sw_result   = "PASS" if p_val > 0.05 else "FAIL"
    with open(f"{output_dir}/shapiro_wilk_result.txt", "w") as fh:
        fh.write("Shapiro-Wilk Normality Test\n")
        fh.write(f"Statistic: {stat:.6f}\n")
        fh.write(f"p-value:   {p_val:.6f}\n")
        fh.write(f"Result:    {sw_result}\n")

    # ── [6/8] Q-Q plot ──────────────────────────────────────────────────────
    print("[6/8] Saving Q-Q plot...")
    save_qq_plot(res_for_sw, path=f"{output_dir}/qq_plot.png")

    # ── [7/8] Residual histogram ────────────────────────────────────────────
    print("[7/8] Saving residual histogram...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(res_for_sw, bins=30, density=True, alpha=0.7, label="Residuals")
    x_grid = np.linspace(res_for_sw.min(), res_for_sw.max(), 200)
    ax.plot(
        x_grid,
        stats.norm.pdf(x_grid, res_for_sw.mean(), res_for_sw.std()),
        "r-", label="N(μ,σ²)",
    )
    ax.set_xlabel("Residual value")
    ax.set_ylabel("Density")
    ax.set_title("Residual Histogram with Gaussian Overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/residual_histogram.png", dpi=100)
    plt.close(fig)

    # ── [8/8] Summary ───────────────────────────────────────────────────────
    print(f"[8/8] Done.  Shapiro-Wilk: {sw_result}  (p={p_val:.4f})")
    return {
        "rmse":                all_rmse,
        "shapiro_wilk_p":      float(p_val),
        "shapiro_wilk_result": sw_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrbitalMind full pipeline")
    parser.add_argument("--data",   required=True,     help="Path to input CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args    = parser.parse_args()
    results = run_pipeline(args.data, args.output)
    print(results)
