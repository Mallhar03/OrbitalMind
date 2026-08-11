# Skill: Diffusion Model for Residuals
# Iteration 4c

---

## GOAL
Train a 1D diffusion model on prediction residuals — the part of the
signal that LSTM, TFT, and Neural ODE could not capture.
Generates realistic stochastic residual samples.

---

## WHAT RESIDUALS ARE IN THIS CONTEXT
After LSTM/TFT/NeuralODE predict, compute:
    residuals = actual_train_values - predicted_train_values

The diffusion model learns the distribution of these residuals.
At prediction time, it samples from that distribution and adds
the sample to the ensemble prediction.

---

## ARCHITECTURE (CPU-safe, 1D signals)

Forward process (add noise):
    betas = torch.linspace(0.0001, 0.02, n_steps=100)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

Denoising network:
    input:  concat(noisy_residual, timestep_embedding) → size seq_len+1
    Linear(seq_len+1, 128) → ReLU → Linear(128,128) → ReLU → Linear(128, seq_len)
    seq_len = 96

Training:
    batch_size = 16
    epochs     = 50
    lr         = 0.001
    loss       = MSELoss on predicted vs actual noise

---

## FUNCTION SIGNATURES

File: src/orbitalmind/models/diffusion.py
class ResidualDiffusion(nn.Module)
def train_diffusion(residuals_array, device='cpu') -> ResidualDiffusion
def sample_residuals(model, n_samples=1, seq_len=96) -> np.ndarray

---

## CONSTRAINTS
- Input residuals_array shape must be (n_windows, 96)
- Each window = one 24-hour residual sequence
- torch.manual_seed(42) before training
- Output samples must be same scale as input residuals

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_diffusion.py -v
ALL must pass:
- Trains without error
- sample_residuals output shape == (1, 96)
- Mean of sampled residuals is within 2 std of training residual mean
- Std of sampled residuals is within 50% of training residual std

---

## FAILURE MODES
If training loss does not decrease: beta schedule too aggressive — try n_steps=50
If samples look like noise: not enough epochs — increase to 100
