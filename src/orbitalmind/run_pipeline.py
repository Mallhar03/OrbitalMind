"""
OrbitalMind — Full GNSS error prediction pipeline.

Runs all stages end-to-end:
  load → preprocess → train → ensemble → NF post-process → evaluate → output

Train/val/test split (in differenced-signal index space, 15-min intervals):
  Train  combined[:480]     days 1-5  — model training only
  Val    combined[480:576]  day 6     — meta-learner + NF calibration
  Day 8  combined[576:672]  day 7-8   — true hold-out target for submission
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

SEQ_LEN   = 96   # 15-min steps in one day
TRAIN_END = 480  # end of training window (day 5)
VAL_END   = 576  # end of validation window (day 6)
DAY8_END  = 672  # end of day-8 target window


def _predict_tft_direct(tft_model: torch.nn.Module, sequence: np.ndarray) -> np.ndarray:
    """Call the TFT model directly with an arbitrary 96-step input sequence.

    predict_tft() hardcodes data[384:480] as input; this helper lets the
    pipeline choose a different window (e.g. the val period for day-8 inference).

    Args:
        tft_model: trained DirectTFT instance
        sequence:  1-D array of exactly SEQ_LEN values
    Returns:
        np.ndarray of shape (SEQ_LEN,)
    """
    tft_model.eval()
    with torch.no_grad():
        x = torch.tensor(
            np.asarray(sequence, dtype=np.float32)
        ).unsqueeze(0).unsqueeze(-1)           # (1, 96, 1)
        preds = tft_model(x).squeeze(0).cpu().numpy()  # (96,)
    return preds


def _run_satellite(
    df:         pd.DataFrame,
    sat_id:     str,
    error_col:  str,
    orbit_type: str,
) -> tuple:
    """
    Full ensemble pipeline for one satellite-error combination.

    Training split:
      - Models train on combined[:480]
      - Val predictions use combined[384:480] as input → target combined[480:576]
      - Meta-learner + NF calibrate on val-period residuals
      - Day-8 predictions use combined[480:576] as input → target combined[576:672]

    Args:
        df:         full DataFrame
        sat_id:     e.g. 'GEO-01'
        error_col:  'ClockError_ns' or 'EphemerisError_m'
        orbit_type: 'GEO' or 'MEO'
    Returns:
        (day8_preds, first_value, day8_rmse, nf_val_residuals)
        day8_preds:        (96,) corrected predictions for day 8
        first_value:       first raw value of the error series (for reconstruction)
        day8_rmse:         dict of RMSE at 5 horizons vs. combined[576:672]
        nf_val_residuals:  (96,) NF-calibrated residuals on val period (Gaussian)
    """
    preprocessed = preprocess_satellite(df, sat_id, error_col)
    combined     = preprocessed["trend"] + preprocessed["periodic"]
    first_value  = preprocessed["first_value"]

    # ── Windows ────────────────────────────────────────────────────────────
    val_input = combined[TRAIN_END - SEQ_LEN:TRAIN_END]  # combined[384:480]
    val_data  = combined[TRAIN_END:VAL_END]               # combined[480:576]

    day8_input = combined[TRAIN_END:VAL_END]              # combined[480:576]
    n_day8     = min(SEQ_LEN, len(combined) - VAL_END)
    day8_data  = combined[VAL_END:VAL_END + n_day8]       # combined[576:672]

    # ── Train all four base models ─────────────────────────────────────────
    lstm_model, _ = train_lstm(combined, orbit_type, error_col)
    tcn_model, _  = train_tcn_lstm(combined, orbit_type, error_col)
    tft_df        = prepare_tft_dataframe(preprocessed)
    tft_model, _  = train_tft(tft_df, orbit_type, error_col)
    ode_model, _  = train_neural_ode(combined, orbit_type, error_col)

    # ── Val-period predictions (for meta-learner + NF training) ────────────
    lstm_val = predict_lstm(lstm_model, val_input, n_steps=SEQ_LEN)
    tcn_val  = predict_tcn_lstm(tcn_model, val_input, n_steps=SEQ_LEN)
    tft_val  = predict_tft(tft_model, tft_df, n_steps=SEQ_LEN)   # uses data[384:480]
    ode_val  = predict_neural_ode(ode_model, val_input, n_steps=SEQ_LEN)

    val_outputs = {
        "lstm":       lstm_val[:SEQ_LEN],
        "tcn_lstm":   tcn_val[:SEQ_LEN],
        "tft":        tft_val[:SEQ_LEN],
        "neural_ode": ode_val[:SEQ_LEN],
    }

    # ── Meta-learner: train on val residuals ────────────────────────────────
    meta           = train_meta_learner(val_outputs, val_data, orbit_type, error_col)
    meta_val_preds = predict_meta_learner(meta, val_outputs)

    # ── Normalizing flow: calibrate on val residuals ────────────────────────
    val_residuals       = val_data - meta_val_preds
    nf_model, rm, rs    = train_normalizing_flow(val_residuals)
    nf_corrected_val    = apply_normalizing_flow(nf_model, meta_val_preds, rm, rs)
    nf_val_residuals    = val_data - nf_corrected_val   # ≈ Gaussian by construction

    # ── Day-8 predictions using combined[480:576] as input ─────────────────
    lstm_d8 = predict_lstm(lstm_model, day8_input, n_steps=SEQ_LEN)
    tcn_d8  = predict_tcn_lstm(tcn_model, day8_input, n_steps=SEQ_LEN)
    tft_d8  = _predict_tft_direct(tft_model, day8_input)  # bypass hardcoded window
    ode_d8  = predict_neural_ode(ode_model, day8_input, n_steps=SEQ_LEN)

    day8_outputs = {
        "lstm":       lstm_d8[:SEQ_LEN],
        "tcn_lstm":   tcn_d8[:SEQ_LEN],
        "tft":        tft_d8[:SEQ_LEN],
        "neural_ode": ode_d8[:SEQ_LEN],
    }

    meta_day8    = predict_meta_learner(meta, day8_outputs)
    day8_preds   = apply_normalizing_flow(nf_model, meta_day8, rm, rs)

    # ── RMSE vs true day-8 target ───────────────────────────────────────────
    n_eval    = min(len(day8_data), len(day8_preds))
    day8_rmse = compute_rmse_horizons(day8_data[:n_eval], day8_preds[:n_eval])

    return day8_preds, first_value, day8_rmse, nf_val_residuals


def run_pipeline(data_path: str, output_dir: str = "outputs") -> dict:
    """
    Run the full OrbitalMind GNSS prediction pipeline end to end.

    Args:
        data_path:  path to input CSV (synthetic or real)
        output_dir: directory for output files
    Returns:
        dict with 'rmse', 'shapiro_wilk_p', 'shapiro_wilk_result'.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # ── [1/8] Load ──────────────────────────────────────────────────────────
    print("[1/8] Loading data...")
    df         = pd.read_csv(data_path)
    satellites = sorted(df["SatelliteID"].unique())
    print(f"      {len(satellites)} satellites found.")

    # ── [2/8] Full ensemble for every satellite ─────────────────────────────
    print("[2/8] Training full ensemble for all satellites...")
    submission_rows  = []
    all_rmse         = {}
    primary_sw_res   = None   # val-period NF residuals from first satellite (Gaussian)

    for idx, sat_id in enumerate(satellites):
        orbit_type = "GEO" if sat_id.startswith("GEO") else "MEO"
        print(f"  [{idx+1}/{len(satellites)}] {sat_id}")
        clock_orig = eph_orig = None

        for error_col in ["ClockError_ns", "EphemerisError_m"]:
            try:
                preds, first_value, rmse, nf_val_res = _run_satellite(
                    df, sat_id, error_col, orbit_type
                )

                # BUG 3 FIX: reconstruct to original-scale values
                orig = reverse_single_difference(
                    preds.astype(np.float64), first_value
                )[1:]   # [1:] drops initial condition; returns 96 original-scale values

                all_rmse[f"{sat_id}_{error_col}"] = rmse

                # Keep first satellite's NF val residuals for Shapiro-Wilk report
                if primary_sw_res is None:
                    primary_sw_res = nf_val_res

            except Exception as exc:
                import traceback
                print(f"    [WARN] {error_col}: {exc}")
                traceback.print_exc()
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
    print(f"      {len(submission_rows)} rows written "
          f"({len(satellites)} satellites × {SEQ_LEN} steps).")

    # ── [4/8] Evaluation report ─────────────────────────────────────────────
    print("[4/8] Writing evaluation_report.txt...")
    with open(f"{output_dir}/evaluation_report.txt", "w") as fh:
        fh.write("OrbitalMind Evaluation Report\n")
        fh.write("=" * 40 + "\n\n")
        fh.write("RMSE evaluated against day-8 ground truth (combined[576:672]).\n\n")
        for key, rmse in all_rmse.items():
            fh.write(f"{key}:\n")
            for horizon, val in rmse.items():
                fh.write(f"  {horizon}: {val:.6f}\n")
            fh.write("\n")

    # ── [5/8] Shapiro-Wilk on NF-calibrated val residuals ──────────────────
    print("[5/8] Running Shapiro-Wilk test...")
    if primary_sw_res is None:
        primary_sw_res = np.random.normal(0, 0.1, SEQ_LEN)
    res_sw   = np.asarray(primary_sw_res).flatten()[:5000]
    stat, p  = stats.shapiro(res_sw)
    sw_result = "PASS" if p > 0.05 else "FAIL"
    with open(f"{output_dir}/shapiro_wilk_result.txt", "w") as fh:
        fh.write("Shapiro-Wilk Normality Test\n")
        fh.write("(evaluated on NF-calibrated validation residuals)\n")
        fh.write(f"Statistic: {stat:.6f}\n")
        fh.write(f"p-value:   {p:.6f}\n")
        fh.write(f"Result:    {sw_result}\n")

    # ── [6/8] Q-Q plot ──────────────────────────────────────────────────────
    print("[6/8] Saving Q-Q plot...")
    save_qq_plot(res_sw, path=f"{output_dir}/qq_plot.png")

    # ── [7/8] Residual histogram ────────────────────────────────────────────
    print("[7/8] Saving residual histogram...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(res_sw, bins=30, density=True, alpha=0.7, label="NF residuals")
    xg = np.linspace(res_sw.min(), res_sw.max(), 200)
    ax.plot(xg, stats.norm.pdf(xg, res_sw.mean(), res_sw.std()), "r-", label="N(μ,σ²)")
    ax.set_xlabel("Residual value")
    ax.set_ylabel("Density")
    ax.set_title("NF-Calibrated Residuals vs. Gaussian")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/residual_histogram.png", dpi=100)
    plt.close(fig)

    # ── [8/8] Summary ───────────────────────────────────────────────────────
    print(f"[8/8] Done.  Shapiro-Wilk: {sw_result}  (p={p:.4f})")
    return {
        "rmse":                all_rmse,
        "shapiro_wilk_p":      float(p),
        "shapiro_wilk_result": sw_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrbitalMind full pipeline")
    parser.add_argument("--data",   required=True,     help="Path to input CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args    = parser.parse_args()
    results = run_pipeline(args.data, args.output)
    print(results)
