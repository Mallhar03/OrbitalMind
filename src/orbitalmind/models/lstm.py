"""LSTM sequence model for GNSS satellite error prediction."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from orbitalmind.paths import models_dir

SEQ_LEN    = 96
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.2
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 0.001
SAVE_DIR   = models_dir()


def _make_sequences(data: np.ndarray, seq_len: int):
    """Sliding-window sequence builder for next-step prediction."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMPredictor(nn.Module):
    """Two-layer LSTM for univariate time series."""

    def __init__(self, input_size=1, hidden_size=HIDDEN, num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1)
        Returns:
            (batch,) predictions
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_lstm(
    data_array: np.ndarray,
    orbit_type: str,
    error_col: str,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Train an LSTMPredictor on the given satellite error signal.

    Args:
        data_array: 1-D array of combined (trend + periodic) signal values
        orbit_type: 'GEO' or 'MEO' — used for model filename
        error_col: 'ClockError_ns' or 'EphemerisError_m'
        device: torch device string (always 'cpu' for local runs)
    Returns:
        (trained model, metrics dict with initial_train_loss and final_train_loss)
    """
    torch.manual_seed(42)
    dev = torch.device(device)

    train_data = np.asarray(data_array, dtype=np.float32)
    X_np, y_np = _make_sequences(train_data, SEQ_LEN)
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
    torch.save(model.state_dict(), f"{SAVE_DIR}/lstm_{orbit_type}_{error_col}.pt")

    return model, {"initial_train_loss": initial_loss, "final_train_loss": final_loss}


def predict_lstm(
    model: nn.Module,
    last_sequence: np.ndarray,
    n_steps: int = 96,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate n_steps ahead predictions via autoregressive rollout.

    Args:
        model: trained LSTMPredictor
        last_sequence: 1-D array of the most recent SEQ_LEN values
        n_steps: number of future steps to predict
        device: torch device string
    Returns:
        np.ndarray of shape (n_steps,) in original signal units.
    """
    dev = torch.device(device)
    model.eval()
    seq = list(np.asarray(last_sequence, dtype=np.float32)[-SEQ_LEN:])
    preds = []

    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(dev)
            val = model(x).item()
            preds.append(val)
            seq.append(val)

    return np.array(preds, dtype=np.float32)
