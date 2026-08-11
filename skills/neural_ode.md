# Skill: Neural ODE Model
# Iteration 4b (runs parallel to LSTM iteration)

---

## GOAL
Model satellite clock drift as a continuous physical process
using Neural Ordinary Differential Equations.
Instead of predicting the next value, predict the rate of change.

---

## WHY NEURAL ODE FOR THIS PROBLEM
Clock drift is physically continuous — it does not jump discretely
every 15 minutes. A Neural ODE models dh/dt = f(h,t) where h is
the hidden state. Integrating f over time gives smooth, physically
correct predictions of how the error evolves.

---

## LIBRARY
torchdiffeq — install: pip install torchdiffeq

---

## ARCHITECTURE (CPU-safe)

ODEFunc (the derivative network):
    input:  hidden_size = 32
    layers: Linear(32,64) → Tanh → Linear(64,64) → Tanh → Linear(64,32)
    output: hidden_size = 32

NeuralODEPredictor:
    Encoder: LSTM(input=1, hidden=32, layers=1) → produces h0
    ODE:     odeint(ode_func, h0, t_span)
    Decoder: Linear(32, 1)

t_span: torch.linspace(0, 1, 96) — normalised time for 96 prediction steps

Training config (CPU-safe):
    batch_size     = 8    # small — ODE solve is expensive on CPU
    epochs         = 20
    learning_rate  = 0.001
    gradient clip  = 1.0  # required — ODE gradients can explode

---

## FUNCTION SIGNATURES

File: src/orbitalmind/models/neural_ode.py
class ODEFunc(nn.Module)
class NeuralODEPredictor(nn.Module)
def train_neural_ode(data_array, orbit_type, error_col, device='cpu') -> tuple[nn.Module, dict]
def predict_neural_ode(model, last_sequence, n_steps=96, device='cpu') -> np.ndarray

---

## CONSTRAINTS
- gradient_clip_val = 1.0 is mandatory — without it ODE training diverges
- batch_size = 8 maximum on CPU — ODE solver runs forward pass multiple times
- odeint method = 'rk4' (fixed step) — faster than adaptive on CPU
- torch.manual_seed(42) before training

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_neural_ode.py -v
ALL must pass:
- Trains without NaN loss
- Output shape == (96,)
- Predictions are smooth (no sudden jumps > 5 ns between consecutive steps)
- RMSE at 1hr < 2.5 ns (can be worse than LSTM — it contributes to ensemble)

---

## FAILURE MODES
If NaN loss immediately: gradient exploding — reduce lr to 0.0001, clip to 0.5
If training too slow on CPU: reduce epochs to 10 for local verify
If odeint fails: switch method from 'rk4' to 'euler' as fallback
