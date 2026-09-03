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
