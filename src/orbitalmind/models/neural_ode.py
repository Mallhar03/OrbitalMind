"""Neural ODE model for physics-consistent GNSS satellite error prediction.

ODE TIME AND REAL TIME
----------------------
The integration variable `t` is neither real time nor the step index. One ODE
time unit is defined here as exactly TRAIN_STEPS real steps of the series, i.e.
TRAIN_STEPS x 15 minutes. The grid is therefore

    t_span = [1, 2, ..., n_steps] / TRAIN_STEPS

so step k always sits at t = k / TRAIN_STEPS whatever n_steps is, training ends
at t = 1.0, and a 96-step forecast reaches t = 96 / TRAIN_STEPS.

It is worth stating why, because the obvious-looking alternative was wrong and
shipped for a while. The grid used to be `torch.linspace(0.0, 1.0, n_steps + 1)[1:]`,
which spans (0, 1] *regardless of n_steps*. n_steps then changed only where the
trajectory was sampled, never how far it was integrated. Training ran with
n_steps=TRAIN_STEPS=8 against targets 8 steps ahead, so the network learned
"t = 1.0 means +8 real steps" = 2 hours; inference ran with n_steps=96 and
consumed that same t = 1.0 as the +96-step, 24-hour prediction. The 24-hour
forecast was a two-hour trajectory resampled on a 12x finer grid. With the grid
above, ODE time and real time agree in training and inference by construction.

Note that torchdiffeq anchors y0 at t_span[0], so the encoder state is the state
at the first forecast step and the solver integrates over t_span[0] .. t_span[-1],
a span of (n_steps - 1) / TRAIN_STEPS. Training and inference share the same step
spacing, 1 / TRAIN_STEPS, so the fixed-step rk4 discretisation the dynamics were
fitted under is the one they are rolled out under.

WHY TRAIN_STEPS IS 16
---------------------
TRAIN_STEPS sets the training horizon and therefore how far inference has to
extrapolate: it integrates to t = 96 / TRAIN_STEPS, and everything past t = 1.0
is outside the range the dynamics were ever fitted on. TRAIN_STEPS=96 would
remove extrapolation entirely, and TRAIN_STEPS=8 is the cheapest. Both were
measured on real GNSS series (seven satellite/column series, training on the
backtest window and scoring the 96 steps that follow it -- a window disjoint from
the reported backtest target and from the held-out day), and neither is the right
answer:

  * TRAIN_STEPS=8 extrapolates 13.6x. It never diverged -- the derivative net
    reads a tanh-bounded layer, so |dh/dt| is globally bounded and the state can
    drift at most linearly -- but the drift alone was enough. Forecast amplitude
    reached 0.55-0.84x the training signal's own maximum on several satellites,
    and 24-hour RMSE came out 6% to 187% worse than a flat zero forecast on all
    seven series. Bounded is not the same as usable.
  * TRAIN_STEPS=96 costs 8-9.5x the training wall-clock of TRAIN_STEPS=8, which
    on a 95-satellite run (four ODE fits per satellite: two error columns x two
    plans) turns roughly one hour of the pipeline into roughly eight. It also
    collapses: fitted against 96-step targets the network learns that the series
    mean is the best long-horizon guess, and its forecasts matched the zero
    baseline to three decimal places at every horizon. Nine times the cost for a
    constant.
  * TRAIN_STEPS=24 and 32 collapse the same way at 2.3-3x the cost.

TRAIN_STEPS=16 is the compromise the measurements support: about 2.4x the cost
of 8, still enough signal to beat a zero forecast at the 1-6 hour horizons on
most satellites, and contained enough that 24-hour RMSE stayed within a few
percent of the zero baseline on five of the seven series. Its residual risk is
the 6.3x extrapolation, which the non-finite guard below turns into a visible
failure rather than NaNs in the ensemble.

Do not change TRAIN_STEPS without re-measuring the 96-step backtest: lowering it
does not merely shorten training, it silently pushes inference further into
extrapolation, and that is where the damage was.
"""
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

from orbitalmind.paths import models_dir

from orbitalmind.device import resolve_device

SEQ_LEN       = 96
HIDDEN_SIZE   = 32
BATCH_SIZE    = 8
EPOCHS        = 20
LR            = 0.001
GRAD_CLIP     = 1.0
# ODE steps trained on, and the definition of one ODE time unit: t = 1.0 is
# TRAIN_STEPS real steps ahead, here 16 x 15 min = 4 hours. Chosen on measured
# cost and 96-step backtest accuracy -- see the module docstring.
TRAIN_STEPS   = 16
SAVE_DIR = models_dir()


class ODEDivergedError(RuntimeError):
    """Raised when the integrated trajectory leaves the finite range.

    The pipeline catches exceptions per satellite and records them as visible
    fallbacks, so raising is how a diverged solve reaches the caller. Returning
    the NaNs instead would poison the ensemble silently.
    """


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

    def __init__(self, hidden_size: int = HIDDEN_SIZE, time_scale: int | None = None):
        """
        Args:
            hidden_size: width of the encoder hidden state and ODE state
            time_scale:  real steps per ODE time unit; defaults to TRAIN_STEPS.
                         Held on the instance so a model always integrates on
                         the grid it was trained on, whatever n_steps a caller
                         asks for.
        """
        super().__init__()
        self.encoder  = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.ode_func = ODEFunc(hidden_size)
        self.decoder  = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.time_scale  = int(time_scale if time_scale is not None else TRAIN_STEPS)

    def forward(self, x: torch.Tensor, n_steps: int = TRAIN_STEPS) -> torch.Tensor:
        """
        Encode input sequence, solve ODE, decode trajectory.

        Args:
            x: (batch, seq_len, 1) input sequence
            n_steps: number of ODE trajectory points to generate
        Returns:
            (batch, n_steps) predictions
        Raises:
            ODEDivergedError: if the integrated trajectory is not finite.
        """
        _, (h_n, _) = self.encoder(x)
        h0       = h_n[-1]                                  # (batch, hidden)
        # One ODE time unit = self.time_scale real steps, so step k sits at
        # t = k / time_scale and the grid spacing is the same in training and in
        # inference. odeint treats t_span[0] as the initial time, so h0 is the
        # state at the first forecast step and preds[:, 0] is that state decoded.
        t_span   = torch.arange(1, n_steps + 1, dtype=torch.float32,
                                device=x.device) / float(self.time_scale)
        h_traj   = odeint(self.ode_func, h0, t_span, method="rk4")  # (n_steps, batch, hidden)
        if not torch.isfinite(h_traj).all():
            raise ODEDivergedError(
                f"Neural ODE trajectory left the finite range: n_steps={n_steps}, "
                f"time_scale={self.time_scale}, integrated to t={n_steps / self.time_scale:.3f}"
            )
        preds    = self.decoder(h_traj).squeeze(-1)         # (n_steps, batch)
        return preds.permute(1, 0)                          # (batch, n_steps)


def train_neural_ode(
    data_array: np.ndarray,
    orbit_type: str,
    error_col: str,
    device: str | None = None,
    model_tag: str | None = None,
) -> tuple[nn.Module, dict]:
    """
    Train a NeuralODEPredictor on the given satellite error signal.

    Args:
        data_array: 1-D combined (trend + periodic) signal array
        orbit_type: 'GEO' or 'MEO'
        model_tag:  identifier for the saved weights (satellite id when available)
        error_col: 'ClockError_ns' or 'EphemerisError_m'
        device: torch device string, or None to resolve automatically
    Returns:
        (trained model, metrics dict with initial_train_loss, final_train_loss
         and diverged_batches)
    Raises:
        ODEDivergedError: if training produced a non-finite loss, or if no batch
            in an epoch integrated successfully. A model in that state would feed
            NaNs into the ensemble, so it is refused rather than returned.
    """
    torch.manual_seed(42)
    dev = resolve_device(device)

    train_data = np.asarray(data_array, dtype=np.float32)
    X_np, y_np = _make_sequences(train_data, SEQ_LEN, TRAIN_STEPS)
    X_t = torch.tensor(X_np).unsqueeze(-1)   # (n, seq_len, 1)
    y_t = torch.tensor(y_np)                  # (n, TRAIN_STEPS)

    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model   = NeuralODEPredictor().to(dev)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    initial_loss = final_loss = None
    diverged_batches = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_ok = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            try:
                pred = model(xb, n_steps=TRAIN_STEPS)   # (batch, TRAIN_STEPS)
            except ODEDivergedError:
                # One unstable batch should not abandon the run, but it must not
                # pass unnoticed either: it is counted, warned about, and the
                # epoch fails outright if nothing integrates.
                diverged_batches += 1
                continue
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_loss += loss.item()
            n_ok += 1
        if n_ok == 0:
            raise ODEDivergedError(
                f"every batch diverged in epoch {epoch} "
                f"({model_tag or orbit_type}, {error_col})"
            )
        epoch_loss /= n_ok
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    if diverged_batches:
        warnings.warn(
            f"Neural ODE ({model_tag or orbit_type}, {error_col}): "
            f"{diverged_batches} batches skipped after a non-finite solve",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.isfinite(final_loss):
        raise ODEDivergedError(
            f"Neural ODE training loss is not finite ({model_tag or orbit_type}, "
            f"{error_col}): final_train_loss={final_loss}"
        )

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Sanitize error_col for use in a filename: 'satclockerror (m)' -> 'satclockerror_m'
    col_tag = error_col.replace(" ", "_").replace("(", "").replace(")", "")
    torch.save(model.state_dict(), f"{SAVE_DIR}/neural_ode_{model_tag or orbit_type}_{col_tag}.pt")

    return model, {"initial_train_loss": float(initial_loss),
                   "final_train_loss": float(final_loss),
                   "diverged_batches": diverged_batches}


def predict_neural_ode(
    model: nn.Module,
    last_sequence: np.ndarray,
    n_steps: int = 96,
    device: str | None = None,
) -> np.ndarray:
    """
    Generate n_steps smooth predictions using the trained Neural ODE.

    Args:
        model: trained NeuralODEPredictor
        last_sequence: 1-D array of the most recent SEQ_LEN values
        n_steps: number of future steps to predict
        device: torch device string, or None to resolve automatically
    Returns:
        np.ndarray of shape (n_steps,) — smooth, physically consistent predictions.
    Raises:
        ODEDivergedError: if the solve or the decoded forecast is not finite.
            Raising is deliberate: run_pipeline catches per-satellite exceptions
            and records them as visible fallbacks, whereas an array of NaNs would
            travel silently into the meta-learner and the ensemble.
    """
    dev = resolve_device(device)
    model.eval()
    seq = np.asarray(last_sequence, dtype=np.float32)[-SEQ_LEN:]
    x   = torch.tensor(seq).unsqueeze(0).unsqueeze(-1).to(dev)  # (1, seq_len, 1)

    with torch.no_grad():
        preds = model(x, n_steps=n_steps)   # (1, n_steps)

    out = preds.squeeze(0).cpu().numpy()
    if not np.all(np.isfinite(out)):
        raise ODEDivergedError(
            f"Neural ODE forecast is not finite over {n_steps} steps "
            f"({int(np.sum(~np.isfinite(out)))} of {out.size} values)"
        )
    return out
