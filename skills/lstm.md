# Skill: LSTM + TCN-LSTM Models
# Iteration 4

---

## GOAL
Build and train two sequence models — LSTM and TCN-LSTM.
Train separate models for GEO and MEO orbit types.
Generate 96-step Day-8 predictions for both ClockError_ns and EphemerisError_m.

---

## INPUT
Preprocessed data dict from iteration 2.
Feature matrix from iteration 3.
Device: CPU (no local GPU — keep models small)

## OUTPUT
For each satellite, for each error type:
    predictions: np.ndarray of shape (96,)
    rmse_per_horizon: dict with keys 15min, 30min, 1hr, 2hr, 24hr

---

## MODEL 1: LSTM

Architecture:
    input_size  = 1 (univariate — one error signal at a time)
    hidden_size = 64 (small for CPU — do not increase)
    num_layers  = 2
    dropout     = 0.2
    output_size = 1

Training config (CPU-safe):
    sequence_length = 96
    batch_size      = 16
    epochs          = 30
    learning_rate   = 0.001
    optimizer       = Adam
    loss            = MSELoss
    train split     = whatever array the caller passes in.
                      train_lstm() no longer slices its own data;
                      orbitalmind.splits owns every window.

Prediction method: rolling — each prediction feeds back as input for next step

## MODEL 2: TCN-LSTM

Architecture:
    TCN part:
        num_channels = [32, 64]     # small for CPU
        kernel_size  = 3
        dilation     = [1, 2]       # two layers only
    LSTM part:
        hidden_size  = 64
        num_layers   = 1
    Combined output: linear(64, 1)

Training config: identical to LSTM above

---

## FUNCTION SIGNATURES

File: src/orbitalmind/models/lstm.py
class LSTMPredictor(nn.Module)
def train_lstm(data_array, orbit_type, error_col, device='cpu') -> tuple[nn.Module, dict]
def predict_lstm(model, last_sequence, n_steps=96, device='cpu') -> np.ndarray

File: src/orbitalmind/models/tcn_lstm.py
class TCNBlock(nn.Module)
class TCNLSTMPredictor(nn.Module)
def train_tcn_lstm(data_array, orbit_type, error_col, device='cpu') -> tuple[nn.Module, dict]
def predict_tcn_lstm(model, last_sequence, n_steps=96, device='cpu') -> np.ndarray

File: src/orbitalmind/models/base_trainer.py
def compute_rmse_horizons(y_true, y_pred) -> dict
    # returns {15min: float, 30min: float, 1hr: float, 2hr: float, 24hr: float}
    # horizons map to steps: 1, 2, 4, 8, 96

---

## CONSTRAINTS
- device = torch.device('cpu') — hardcoded for local runs
- torch.manual_seed(42) at top of every train function
- Train GEO satellites and MEO satellites with separate model instances
- Save trained model weights to: models/saved/lstm_GEO_clock.pt etc
- Never train on day 8 data

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_lstm.py -v
ALL must pass:
- Model trains without error (loss decreases over epochs)
- Output predictions shape == (96,)
- RMSE at 15min < 1.5 ns on synthetic data
- RMSE at 1hr < 2.0 ns on synthetic data
- RMSE at 24hr < 5.0 ns on synthetic data
- GEO model and MEO model are separate saved files
- Predictions are in original scale (not normalised)

---

## FAILURE MODES
If loss does not decrease: learning rate too high — try 0.0001
If predictions collapse to mean: sequence_length too long for data size — try 48
If memory error: batch_size too large — reduce to 8
