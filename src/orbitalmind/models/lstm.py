"""LSTM sequence model for GNSS satellite error prediction."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from orbitalmind.paths import models_dir

from orbitalmind.device import resolve_device

SEQ_LEN    = 96
PRED_LEN   = 96
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.2
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 0.001
SAVE_DIR   = models_dir()


def _make_sequences(data: np.ndarray, seq_len: int, pred_len: int):
    """Sliding-window sequence builder for direct multi-step prediction."""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMPredictor(nn.Module):
    """Two-layer LSTM for univariate time series."""

    def __init__(self, input_size=1, hidden_size=HIDDEN, num_layers=N_LAYERS, dropout=DROPOUT, pred_len=PRED_LEN):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1)
        Returns:
            (batch, pred_len) predictions
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(
    data_array: np.ndarray,
    orbit_type: str,
    error_col: str,
    device: str | None = None,
    model_tag: str | None = None,
) -> tuple[nn.Module, dict]:
    """
    Train an LSTMPredictor on the given satellite error signal.

    Args:
        data_array: 1-D array of combined (trend + periodic) signal values
        orbit_type: 'GEO' or 'MEO'
        model_tag:  identifier for the saved weights. Models are trained PER
                    SATELLITE, so a filename keyed only on orbit type makes every
                    satellite of that type overwrite the previous one and leaves
                    only the last one's weights on disk.
        error_col: 'ClockError_ns' or 'EphemerisError_m'
        device: torch device string, or None to resolve automatically
    Returns:
        (trained model, metrics dict with initial_train_loss and final_train_loss)
    """
    torch.manual_seed(42)
    dev = resolve_device(device)

    train_data = np.asarray(data_array, dtype=np.float32)
    X_np, y_np = _make_sequences(train_data, SEQ_LEN, PRED_LEN)
    X_t = torch.tensor(X_np).unsqueeze(-1)  # (n, seq, 1)
    y_t = torch.tensor(y_np)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMPredictor().to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    initial_loss = final_loss = None
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/lstm_{model_tag or orbit_type}_{error_col}.pt")

    return model, {"initial_train_loss": initial_loss, "final_train_loss": final_loss}


def predict_lstm(
    model: nn.Module,
    last_sequence: np.ndarray,
    n_steps: int = 96,
    device: str | None = None,
) -> np.ndarray:
    """
    Generate direct multi-step predictions.

    Args:
        model: trained LSTMPredictor
        last_sequence: 1-D array of the most recent SEQ_LEN values
        n_steps: number of future steps to predict (must be <= PRED_LEN)
        device: torch device string, or None to resolve automatically
    Returns:
        np.ndarray of shape (n_steps,) in original signal units.
    """
    dev = resolve_device(device)
    model.eval()
    seq = np.asarray(last_sequence, dtype=np.float32).flatten()
    if len(seq) < SEQ_LEN:
        raise ValueError(f"predict_lstm needs {SEQ_LEN} input steps, got {len(seq)}")
    seq = seq[-SEQ_LEN:]

    with torch.no_grad():
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(dev)
        preds = model(x).squeeze(0).cpu().numpy()

    return preds[:n_steps]
