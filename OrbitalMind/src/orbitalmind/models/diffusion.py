"""
Simple diffusion-style generative model for GNSS prediction residuals.

Models the residual distribution as a Gaussian and samples from it.
Serves as a lightweight alternative to score-based diffusion for CPU runs.
"""
import numpy as np


class _ResidualDiffusionModel:
    """Parametric Gaussian model fitted to residual statistics."""

    def __init__(self, mean: float, std: float, seq_len: int):
        self.mean    = mean
        self.std     = std
        self.seq_len = seq_len

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the learned residual distribution.

        Args:
            n_samples: number of independent sequences to generate
        Returns:
            np.ndarray of shape (n_samples, seq_len)
        """
        return np.random.normal(self.mean, self.std, (n_samples, self.seq_len)).astype(
            np.float32
        )


def train_diffusion(residuals: np.ndarray) -> _ResidualDiffusionModel:
    """
    Fit a diffusion model to a batch of residual sequences.

    Args:
        residuals: 2-D array of shape (n_windows, seq_len) containing
                   residual time-series windows
    Returns:
        Fitted _ResidualDiffusionModel ready for sampling.
    Raises:
        ValueError: if residuals is not 2-D
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError(
            f"residuals must be 2-D (n_windows, seq_len), got shape {residuals.shape}"
        )
    mean    = float(residuals.mean())
    std     = float(residuals.std()) + 1e-8
    seq_len = residuals.shape[1]
    return _ResidualDiffusionModel(mean=mean, std=std, seq_len=seq_len)


def sample_residuals(
    model:     _ResidualDiffusionModel,
    n_samples: int = 1,
    seq_len:   int = 96,
) -> np.ndarray:
    """
    Sample synthetic residual sequences from the trained model.

    Args:
        model:     fitted model from train_diffusion
        n_samples: number of sequences to generate
        seq_len:   length of each sequence (must match training seq_len)
    Returns:
        np.ndarray of shape (n_samples, seq_len).
    """
    return model.sample(n_samples=n_samples)[:, :seq_len]
