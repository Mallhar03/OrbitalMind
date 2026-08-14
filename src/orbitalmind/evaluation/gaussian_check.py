"""
Gaussian diagnostic utilities for GNSS residual evaluation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def save_qq_plot(residuals: np.ndarray, path: str = "outputs/qq_plot.png") -> None:
    """
    Generate and save a Q-Q plot comparing residuals to a normal distribution.

    Args:
        residuals: 1-D array of residual values to evaluate
        path:      output file path for the saved PNG
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: Residuals vs Normal Distribution")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
    from orbitalmind.preprocessing.pipeline import preprocess_satellite
    from orbitalmind.models.lstm import train_lstm, predict_lstm
    from orbitalmind.models.normalizing_flow import train_normalizing_flow, apply_normalizing_flow

    df          = generate_synthetic_gnss_data(seed=42)
    preprocessed = preprocess_satellite(df, "GEO-01", "ClockError_ns")
    data        = preprocessed["trend"] + preprocessed["periodic"]

    model, _    = train_lstm(data, "GEO", "ClockError_ns")
    val_data    = data[480:576]
    preds       = predict_lstm(model, data[480 - 96:480], n_steps=96)
    residuals   = val_data - preds

    nf_model, res_mean, res_std = train_normalizing_flow(residuals)
    corrected   = apply_normalizing_flow(nf_model, preds, res_mean, res_std)
    final_res   = val_data - corrected

    stat, p = stats.shapiro(final_res)
    print(f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f} ({'PASS' if p > 0.05 else 'FAIL'})")
    save_qq_plot(final_res, path="outputs/qq_plot.png")
    print("Q-Q plot saved to outputs/qq_plot.png")
