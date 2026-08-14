"""TCN-LSTM hybrid model for GNSS satellite error prediction."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN    = 96
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 0.001
SAVE_DIR   = "models/saved"


def _make_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TCNBlock(nn.Module):
    """Temporally Convolutional Block with causal padding and residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_ch, seq_len)
        Returns:
            (batch, out_ch, seq_len) — causal; no future leakage.
        """
        padded = F.pad(x, (self.causal_pad, 0))
        out = self.relu(self.conv(padded))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNLSTMPredictor(nn.Module):
    """TCN feature extractor followed by LSTM temporal modelling."""

    def __init__(self, input_size: int = 1, lstm_hidden: int = 64):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_size, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 64, kernel_size=3, dilation=2),
        )
        self.lstm = nn.LSTM(64, lstm_hidden, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1)
        Returns:
            (batch,) predictions
        """
        # Conv1d needs (batch, channels, seq_len)
        h = x.permute(0, 2, 1)          # (batch, 1, seq_len)
        h = self.tcn(h)                 # (batch, 64, seq_len)
        h = h.permute(0, 2, 1)          # (batch, seq_len, 64)
        out, _ = self.lstm(h)           # (batch, seq_len, hidden)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_tcn_lstm(
    data_array: np.ndarray,
    orbit_type: str,
    error_col: str,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Train a TCNLSTMPredictor on the given satellite error signal.

    Args:
        data_array: 1-D combined (trend + periodic) signal array
        orbit_type: 'GEO' or 'MEO'
        error_col: 'ClockError_ns' or 'EphemerisError_m'
        device: torch device string
    Returns:
        (trained model, metrics dict with initial_train_loss and final_train_loss)
    """
    torch.manual_seed(42)
    dev = torch.device(device)

    train_data = np.asarray(data_array[:480], dtype=np.float32)
    X_np, y_np = _make_sequences(train_data, SEQ_LEN)
    X_t = torch.tensor(X_np).unsqueeze(-1)
    y_t = torch.tensor(y_np)

    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = TCNLSTMPredictor().to(dev)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
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
    torch.save(model.state_dict(), f"{SAVE_DIR}/tcn_lstm_{orbit_type}_{error_col}.pt")

    return model, {"initial_train_loss": initial_loss, "final_train_loss": final_loss}


def predict_tcn_lstm(
    model: nn.Module,
    last_sequence: np.ndarray,
    n_steps: int = 96,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate n_steps predictions via autoregressive rollout.

    Args:
        model: trained TCNLSTMPredictor
        last_sequence: 1-D array of the most recent SEQ_LEN values
        n_steps: number of future steps to predict
        device: torch device string
    Returns:
        np.ndarray of shape (n_steps,).
    """
    dev = torch.device(device)
    model.eval()
    seq   = list(np.asarray(last_sequence, dtype=np.float32)[-SEQ_LEN:])
    preds = []

    with torch.no_grad():
        for _ in range(n_steps):
            x   = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(dev)
            val = model(x).item()
            preds.append(val)
            seq.append(val)

    return np.array(preds, dtype=np.float32)
