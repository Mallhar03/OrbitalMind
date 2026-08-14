"""
Rolling FFT spectral feature extraction for GNSS satellite error series.

Index mapping for a 96-point window at 15-minute spacing (24-hour duration):
  k=1 → period = 24 hours  (GEO dominant frequency)
  k=2 → period = 12 hours  (MEO dominant frequency)
"""
import numpy as np


def create_fft_features(series: np.ndarray, window_size: int = 96) -> np.ndarray:
    """
    Compute sliding-window FFT spectral features for a signal array.

    For each position t ≥ window_size, FFT is computed over series[t-window_size:t].
    Positions before window_size are filled with zeros.

    Features per step:
        amplitude_24hr   — FFT magnitude at k=1 (1 cycle per 24-hour window)
        amplitude_12hr   — FFT magnitude at k=2 (2 cycles per 24-hour window)
        spectral_energy  — total energy of positive-frequency spectrum
        dominant_freq    — index of highest-magnitude frequency bin

    Args:
        series: 1-D array of signal values (length n)
        window_size: FFT window length in samples (default 96 = 24 hours)
    Returns:
        np.ndarray of shape (n, 4) with columns in the order above.
    """
    n = len(series)
    out = np.zeros((n, 4), dtype=float)
    half = window_size // 2
    for t in range(window_size, n):
        window = series[t - window_size:t]
        fft_vals = np.abs(np.fft.fft(window))[:half]
        out[t, 0] = fft_vals[1]               # amplitude_24hr (k=1)
        out[t, 1] = fft_vals[2]               # amplitude_12hr (k=2)
        out[t, 2] = float(np.sum(fft_vals ** 2))  # spectral_energy
        out[t, 3] = float(np.argmax(fft_vals))    # dominant_freq index
    return out
