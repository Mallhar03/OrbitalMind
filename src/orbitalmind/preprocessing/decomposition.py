"""
EMD-based signal decomposition for GNSS error series.

Uses standard EMD (not EEMD) to guarantee exact reconstruction
(sum of IMFs == input signal to floating-point precision).
"""
import numpy as np
from PyEMD import EMD


def decompose_signal(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a differenced GNSS error signal into trend, periodic, and noise components.

    Uses Empirical Mode Decomposition (EMD). Component assignment:
        IMFs[0]      → noise    (highest frequency)
        IMFs[1:-1]   → periodic (mid-range frequencies)
        IMFs[-1]     → trend    (lowest frequency residual)

    sum(trend + periodic + noise) == series exactly (EMD completeness property).

    Args:
        series: 1-D array of differenced satellite error values (length n)
    Returns:
        (trend, periodic, noise) each as np.ndarray of length n.
    """
    np.random.seed(42)
    emd = EMD()
    IMFs = emd.emd(series.astype(float))

    n = len(series)
    zeros = np.zeros(n)

    if IMFs.ndim == 1:
        IMFs = IMFs.reshape(1, -1)

    n_imfs = IMFs.shape[0]

    if n_imfs == 1:
        return IMFs[0].copy(), zeros.copy(), zeros.copy()

    if n_imfs == 2:
        return IMFs[1].copy(), zeros.copy(), IMFs[0].copy()

    noise    = IMFs[0].copy()
    periodic = np.sum(IMFs[1:-1], axis=0)
    trend    = IMFs[-1].copy()
    return trend, periodic, noise
