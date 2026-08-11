# Skill: Temporal Fusion Transformer (TFT)
# Iteration 5

---

## GOAL
Train a Temporal Fusion Transformer for multi-horizon attention.
TFT must outperform LSTM at 24hr horizon.
Train separate models for GEO and MEO.

---

## INPUT
Same preprocessed data and feature matrix as LSTM iteration.
RMSE results from LSTM — TFT must beat LSTM at 24hr horizon.

## OUTPUT
predictions: np.ndarray shape (96,) per satellite per error type
rmse_per_horizon: dict — 24hr RMSE must be lower than LSTM's 24hr RMSE

---

## LIBRARY
pytorch-forecasting==1.1.1
pytorch-lightning==2.1.0
torch==2.1.0

Install: pip install pytorch-forecasting==1.1.1 pytorch-lightning==2.1.0

## ARCHITECTURE (CPU-safe settings)
hidden_size             = 16   # small for CPU
attention_head_size     = 1
dropout                 = 0.1
hidden_continuous_size  = 8
max_encoder_length      = 96
max_prediction_length   = 96

## TRAINING CONFIG
batch_size   = 16
max_epochs   = 20
learning_rate = 0.03
gradient_clip_val = 0.1

---

## DATAFRAME FORMAT REQUIRED BY PYTORCH-FORECASTING
Columns needed:
    time_idx    : int, sequential index starting from 0
    group_id    : str, satellite ID
    target      : float, error value (ClockError_ns or EphemerisError_m)
    time_since_start : float, normalised time feature

---

## FUNCTION SIGNATURES

File: src/orbitalmind/models/tft.py
def prepare_tft_dataframe(preprocessed_data: dict) -> pd.DataFrame
def train_tft(df, orbit_type, error_col, device='cpu') -> tuple[object, dict]
def predict_tft(model, last_known_df, n_steps=96) -> np.ndarray

---

## CONSTRAINTS
- If pytorch-forecasting install fails on Python 3.12, use:
  pip install pytorch-forecasting --pre
- Fallback: if TFT fails to train in 20 epochs, report failure and
  use LSTM predictions for this slot — do NOT block pipeline
- Save model checkpoint to models/saved/tft_GEO_clock.ckpt

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_tft.py -v
ALL must pass:
- TFT trains without error
- Output shape == (96,)
- RMSE at 24hr < LSTM RMSE at 24hr (TFT must beat LSTM at long horizon)
- RMSE at 1hr may be worse than LSTM — that is acceptable
- If TFT fails: fallback flag is set and test still passes with warning

---

## FAILURE MODES
If import error: pytorch-forecasting version conflict — use --pre flag
If NaN loss: reduce learning_rate to 0.003
If OOM on CPU: reduce hidden_size to 8 and batch_size to 8
