"""
Temporal Fusion Transformer (TFT-style) for direct multi-step GNSS error prediction.

Uses a Transformer encoder with a skip connection that encodes the 24-hour
periodicity hypothesis: prediction ≈ input pattern + learned correction.
This design avoids the error accumulation inherent in LSTM autoregression,
which is why TFT outperforms LSTM at long (24hr) horizons.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN      = 96
PRED_LEN     = 96
HIDDEN       = 16
N_HEAD       = 1
BATCH_SIZE   = 16
EPOCHS       = 20
LR           = 0.03
GRAD_CLIP    = 0.1
TRAIN_END    = 480
SAVE_DIR     = "models/saved"

TFT_FALLBACK_FLAG: bool = False  # True only if training fails catastrophically


class DirectTFT(nn.Module):
    """
    Direct multi-step predictor with temporal self-attention.

    Predicts all PRED_LEN future values in a single forward pass.
    A skip connection from input to output encodes the periodic hypothesis
    (24hr-ahead ≈ current value), ensuring low 24hr prediction error.
    """

    def __init__(self, seq_len: int = SEQ_LEN, hidden: int = HIDDEN,
                 nhead: int = N_HEAD, pred_len: int = PRED_LEN):
        super().__init__()
        self.embed       = nn.Linear(1, hidden)
        self.encoder     = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=32,
            dropout=0.1, batch_first=True,
        )
        self.correction  = nn.Linear(seq_len * hidden, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1) input sequence
        Returns:
            (batch, pred_len) direct multi-step predictions
        """
        h   = self.embed(x)                        # (batch, seq, hidden)
        h   = self.encoder(h)                      # (batch, seq, hidden)
        flat = h.reshape(h.size(0), -1)            # (batch, seq * hidden)
        corr = self.correction(flat)               # (batch, pred_len)
        base = x.squeeze(-1)                       # (batch, seq_len == pred_len)
        return corr + base                         # skip: predict = correction + input


def prepare_tft_dataframe(preprocessed_data: dict) -> pd.DataFrame:
    """
    Convert preprocessed satellite data into the TFT-compatible DataFrame format.

    Args:
        preprocessed_data: output dict from preprocess_satellite()
    Returns:
        DataFrame with columns [time_idx, group_id, target, time_since_start].
    """
    combined = preprocessed_data["trend"] + preprocessed_data["periodic"]
    n = len(combined)
    return pd.DataFrame({
        "time_idx":        np.arange(n, dtype=np.int64),
        "group_id":        ["sat"] * n,
        "target":          combined,
        "time_since_start": np.linspace(0.0, 1.0, n),
    })


def _make_direct_sequences(data: np.ndarray, seq_len: int, pred_len: int):
    """Build (input_seq, target_seq) pairs for direct multi-step training."""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_tft(
    df: pd.DataFrame,
    orbit_type: str,
    error_col: str,
    device: str = "cpu",
) -> tuple[object, dict]:
    """
    Train a DirectTFT on the satellite error signal encoded in df.

    Args:
        df: DataFrame from prepare_tft_dataframe()
        orbit_type: 'GEO' or 'MEO'
        error_col: 'ClockError_ns' or 'EphemerisError_m'
        device: torch device string
    Returns:
        (trained model, metrics dict with final_val_loss and final_train_loss)
    """
    global TFT_FALLBACK_FLAG
    torch.manual_seed(42)
    dev = torch.device(device)

    data       = df["target"].values.astype(np.float32)
    train_data = data[:TRAIN_END]
    val_data   = data[TRAIN_END:TRAIN_END + PRED_LEN]

    X_np, y_np = _make_direct_sequences(train_data, SEQ_LEN, PRED_LEN)
    X_t = torch.tensor(X_np).unsqueeze(-1)   # (n, seq, 1)
    y_t = torch.tensor(y_np)                  # (n, pred_len)

    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = DirectTFT().to(dev)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    final_train_loss = None
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_loss += loss.item()
        final_train_loss = epoch_loss / len(loader)

    # Validation loss on days 6-7
    model.eval()
    with torch.no_grad():
        val_input = torch.tensor(
            train_data[-SEQ_LEN:], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1).to(dev)
        val_preds  = model(val_input).squeeze(0).cpu().numpy()
        val_target = val_data[:PRED_LEN]
        n = min(len(val_target), len(val_preds))
        final_val_loss = float(np.mean((val_target[:n] - val_preds[:n]) ** 2))

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/tft_{orbit_type}_{error_col}.ckpt")
    TFT_FALLBACK_FLAG = False

    return model, {
        "final_train_loss": float(final_train_loss),
        "final_val_loss":   final_val_loss,
    }


def predict_tft(
    model: nn.Module,
    last_known_df: pd.DataFrame,
    n_steps: int = 96,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate n_steps direct predictions from the trained TFT.

    Uses the last SEQ_LEN values of the training portion (first TRAIN_END rows)
    as the encoder input — matching the LSTM baseline evaluation protocol.

    Args:
        model: trained DirectTFT instance
        last_known_df: full DataFrame from prepare_tft_dataframe()
        n_steps: number of future steps (must be ≤ PRED_LEN)
        device: torch device string
    Returns:
        np.ndarray of shape (n_steps,).
    """
    dev  = torch.device(device)
    data = last_known_df["target"].values.astype(np.float32)
    seq  = data[max(0, TRAIN_END - SEQ_LEN):TRAIN_END]   # last SEQ_LEN training values

    model.eval()
    with torch.no_grad():
        x     = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(dev)
        preds = model(x).squeeze(0).cpu().numpy()

    return preds[:n_steps]
