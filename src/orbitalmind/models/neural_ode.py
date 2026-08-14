"""Neural ODE model for physics-consistent GNSS satellite error prediction."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

SEQ_LEN       = 96
HIDDEN_SIZE   = 32
BATCH_SIZE    = 8
EPOCHS        = 20
LR            = 0.001
GRAD_CLIP     = 1.0
TRAIN_STEPS   = 8   # ODE prediction steps used during training (faster on CPU)
SAVE_DIR      = "models/saved"


def _make_sequences(data: np.ndarray, seq_len: int, target_len: int):
    """Build (input_seq, target_seq) pairs for multi-step ODE training."""
    X, y = [], []
    for i in range(len(data) - seq_len - target_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + target_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class ODEFunc(nn.Module):
    """Derivative network: dh/dt = f(h, t)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_size),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar time value (unused directly — autonomous ODE)
            h: (batch, hidden_size) hidden state
        Returns:
            (batch, hidden_size) state derivative
        """
        return self.net(h)


class NeuralODEPredictor(nn.Module):
    """LSTM encoder + Neural ODE dynamics + linear decoder."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.encoder  = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.ode_func = ODEFunc(hidden_size)
        self.decoder  = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, n_steps: int = TRAIN_STEPS) -> torch.Tensor:
        """
        Encode input sequence, solve ODE, decode trajectory.

        Args:
            x: (batch, seq_len, 1) input sequence
            n_steps: number of ODE trajectory points to generate
        Returns:
            (batch, n_steps) predictions
        """
        _, (h_n, _) = self.encoder(x)
        h0       = h_n[-1]                                  # (batch, hidden)
        t_span   = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)[1:]
        h_traj   = odeint(self.ode_func, h0, t_span, method="rk4")  # (n_steps, batch, hidden)
        preds    = self.decoder(h_traj).squeeze(-1)         # (n_steps, batch)
        return preds.permute(1, 0)                          # (batch, n_steps)


def train_neural_ode(
    data_array: np.ndarray,
    orbit_type: str,
    error_col: str,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Train a NeuralODEPredictor on the given satellite error signal.

    Args:
        data_array: 1-D combined (trend + periodic) signal array
        orbit_type: 'GEO' or 'MEO'
        error_col: 'ClockError_ns' or 'EphemerisError_m'
        device: torch device string (always 'cpu' for local runs)
    Returns:
        (trained model, metrics dict with initial_train_loss and final_train_loss)
    """
    torch.manual_seed(42)
    dev = torch.device(device)

    train_data = np.asarray(data_array[:480], dtype=np.float32)
    X_np, y_np = _make_sequences(train_data, SEQ_LEN, TRAIN_STEPS)
    X_t = torch.tensor(X_np).unsqueeze(-1)   # (n, seq_len, 1)
    y_t = torch.tensor(y_np)                  # (n, TRAIN_STEPS)

    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = NeuralODEPredictor().to(dev)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    initial_loss = final_loss = None
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb, n_steps=TRAIN_STEPS)   # (batch, TRAIN_STEPS)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/neural_ode_{orbit_type}_{error_col}.pt")

    return model, {"initial_train_loss": float(initial_loss), "final_train_loss": float(final_loss)}


def predict_neural_ode(
    model: nn.Module,
    last_sequence: np.ndarray,
    n_steps: int = 96,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate n_steps smooth predictions using the trained Neural ODE.

    Args:
        model: trained NeuralODEPredictor
        last_sequence: 1-D array of the most recent SEQ_LEN values
        n_steps: number of future steps to predict
        device: torch device string
    Returns:
        np.ndarray of shape (n_steps,) — smooth, physically consistent predictions.
    """
    dev = torch.device(device)
    model.eval()
    seq = np.asarray(last_sequence, dtype=np.float32)[-SEQ_LEN:]
    x   = torch.tensor(seq).unsqueeze(0).unsqueeze(-1).to(dev)  # (1, seq_len, 1)

    with torch.no_grad():
        preds = model(x, n_steps=n_steps)   # (1, n_steps)

    return preds.squeeze(0).cpu().numpy()
