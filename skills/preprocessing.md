# Skill: Preprocessing Pipeline
# Iteration 2

---

## GOAL
Take raw satellite error time series and produce clean, decomposed,
stationary signals ready for model training.

---

## INPUT
DataFrame from synthetic_generator or real hackathon CSV.
Columns: Timestamp, SatelliteID, OrbitType, ClockError_ns, EphemerisError_m

## OUTPUT
Dictionary per satellite per error type:
{
    "sat_id": str,
    "error_col": str,
    "trend": np.ndarray,
    "periodic": np.ndarray,
    "noise": np.ndarray,
    "first_value": float,          # needed to reconstruct original scale
    "original_cleaned": np.ndarray # after outlier removal and IOD correction
    "timestamps": pd.DatetimeIndex
}

---

## FOUR STEPS — IN EXACT ORDER

### Step 1: MAD Outlier Removal
Library: numpy only — no sklearn
Formula:
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    modified_z = 0.6745 * (series - median) / (mad + 1e-8)
    outliers = np.abs(modified_z) > 3.5
    Replace outliers with NaN, then interpolate linearly

### Step 2: IOD Jump Correction
Detection: diff = series.diff().abs()
Threshold: any jump > 2.0 ns or > 1.0 m is an IOD jump
Correction:
    At each jump index, compute offset = value[t] - value[t-1]
    Subtract offset from all values from t onwards
    This makes the series continuous

### Step 3: Single Difference Transformation
    first_value = series.iloc[0]
    differenced = series.diff().dropna()
    Store first_value for reconstruction later

Reconstruction formula (used after prediction):
    reconstructed = differenced.cumsum() + first_value

### Step 4: EWT Signal Decomposition
Library: EMD-signal (PyEMD) — use EEMD class
    from PyEMD import EEMD
    eemd = EEMD()
    eemd.noise_width = 0.05
    IMFs = eemd.eemd(differenced_values)
    noise    = IMFs[0]              # highest frequency
    periodic = np.sum(IMFs[1:-1], axis=0)  # middle IMFs
    trend    = IMFs[-1]             # lowest frequency (residual)

---

## FUNCTION SIGNATURES

File: src/orbitalmind/preprocessing/outlier_removal.py
def remove_outliers_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series

File: src/orbitalmind/preprocessing/iod_correction.py
def correct_iod_jumps(series: pd.Series, threshold_ns: float = 2.0) -> pd.Series

File: src/orbitalmind/preprocessing/differencing.py
def single_difference(series: pd.Series) -> tuple[pd.Series, float]
def reverse_single_difference(differenced: np.ndarray, first_value: float) -> np.ndarray

File: src/orbitalmind/preprocessing/decomposition.py
def decompose_signal(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
# returns (trend, periodic, noise)

File: src/orbitalmind/preprocessing/pipeline.py
def preprocess_satellite(
    df: pd.DataFrame,
    sat_id: str,
    error_col: str
) -> dict
# orchestrates all 4 steps, returns the output dict

---

## CONSTRAINTS
- Process GEO and MEO satellites identically — same pipeline
- Process ClockError_ns and EphemerisError_m separately
- Never modify the original DataFrame — always work on copies
- IOD threshold for EphemerisError_m is 1.0 m (not 2.0 ns)
- EEMD is non-deterministic — set numpy seed 42 before every EEMD call

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_preprocessing.py -v
ALL must pass:
- Output dict has keys: sat_id, error_col, trend, periodic, noise,
  first_value, original_cleaned, timestamps
- trend + periodic + noise reconstructs to within 1e-6 of differenced input
- No NaN in trend, periodic, or noise arrays
- Length of trend == length of original series minus 1 (from differencing)
- IOD jumps in synthetic data are reduced: max(diff(corrected)) < max(diff(original))
- Outlier count after MAD == 0

---

## FAILURE MODES
If EEMD hangs: input signal too short — check series length > 100
If reconstruction error > 1e-6: IMF sum is wrong — check IMFs[1:-1] slice
If IOD correction overshoots: threshold too low — raise to 3.0 temporarily
