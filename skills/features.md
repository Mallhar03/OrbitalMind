# Skill: Feature Engineering
# Iteration 3

---

## GOAL
Build a feature matrix from preprocessed satellite data that captures
lag patterns, rolling statistics, and FFT spectral features.
This matrix is the input to LSTM, TFT, and Neural ODE.

---

## INPUT
Output dict from preprocess_satellite() — specifically trend + periodic arrays

## OUTPUT
X: np.ndarray of shape (n_samples, n_features)
y: np.ndarray of shape (n_samples,)
feature_names: list of strings

---

## FEATURE SET — BUILD ALL OF THESE

### Lag features (most important)
combined = trend + periodic  # reconstruct signal without noise
lags = [1, 2, 4, 8, 16, 32, 96]
For each lag L: feature = combined[t - L]

Why these specific lags:
    lag_1  = 15 min ago
    lag_4  = 1 hour ago
    lag_8  = 2 hours ago
    lag_96 = 24 hours ago (same time yesterday — most predictive for GEO)

### Rolling statistical features
windows = [4, 8, 16, 96]
For each window W:
    rolling_mean_W = mean of last W values
    rolling_std_W  = std of last W values
    rolling_min_W  = min of last W values
    rolling_max_W  = max of last W values

### FFT spectral features
window_size = 96 (= 24 hours)
For each position t, compute FFT over window [t-96:t]:
    fft_vals = np.abs(np.fft.fft(window))[:48]  # positive frequencies only
    amplitude_12hr = fft_vals[8]   # frequency index for 12-hour cycle
    amplitude_24hr = fft_vals[4]   # frequency index for 24-hour cycle
    spectral_energy = np.sum(fft_vals**2)
    dominant_freq   = np.argmax(fft_vals)

### Time-based features
hour_of_day = timestamp.hour (normalised: / 24)
day_of_week = timestamp.dayofweek (normalised: / 6)

### Total feature count
7 lags + 16 rolling (4 windows × 4 stats) + 4 FFT + 2 time = 29 features

---

## FUNCTION SIGNATURES

File: src/orbitalmind/features/lag_features.py
def create_lag_features(series: pd.Series, lags: list = [1,2,4,8,16,32,96]) -> pd.DataFrame

File: src/orbitalmind/features/rolling_features.py
def create_rolling_features(series: pd.Series, windows: list = [4,8,16,96]) -> pd.DataFrame

File: src/orbitalmind/features/fft_features.py
def create_fft_features(series: np.ndarray, window_size: int = 96) -> np.ndarray

File: src/orbitalmind/features/feature_matrix.py
def build_feature_matrix(
    preprocessed_data: dict,
    timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, list]
# returns (X, y, feature_names)

---

## CONSTRAINTS
- Drop all rows with NaN after feature creation (first 96 rows will have NaN)
- Normalise all features to [0,1] range using MinMaxScaler
- Save the scaler object — needed to inverse transform predictions
- feature_names list must match column order in X exactly
- Never leak future information — all features use only past values

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_features.py -v
ALL must pass:
- X shape is (n_samples, 29)
- y shape is (n_samples,)
- No NaN in X or y
- All values in X are between 0 and 1 (normalised)
- amplitude_12hr > amplitude_24hr for MEO satellites
  (MEO has stronger 12hr signal)
- amplitude_24hr > amplitude_12hr for GEO satellites
  (GEO has stronger 24hr signal)
- lag_96 feature has highest correlation with y (> 0.7 for GEO)

---

## FAILURE MODES
If FFT indices wrong: verify window_size=96 and index 4=24hr, 8=12hr
If normalisation breaks: check for zero-variance features and add 1e-8 epsilon
If lag_96 correlation low: signal too noisy — check decomposition output
