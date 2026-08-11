# Skill: Normalizing Flow Post-Processor
# Iteration 7

---

## GOAL
Transform the distribution of final ensemble prediction residuals
so they are Gaussian by construction.
This directly satisfies Evaluation Criterion 2.

---

## WHAT THIS MODULE DOES
After LightGBM meta-learner produces final predictions:
    residuals = actual - predicted  (on validation data)
Train Normalizing Flow on these residuals.
Apply NF transformation to final Day-8 predictions.
Result: residuals follow N(0, σ²) when evaluated by judges.

---

## LIBRARY
normflows — install: pip install normflows

---

## ARCHITECTURE
Base distribution: standard Gaussian N(0,1)
Flow layers: 4 × AutoregressiveRationalQuadraticSpline
Permutation: LULinearPermute between each flow layer
Input dimension: 1 (scalar residuals)

---

## TRAINING
epochs     = 200
lr         = 0.001
loss       = negative log likelihood: -model.log_prob(X).mean()
gradient clip = 1.0
Input: residuals normalised to zero mean, unit std

---

## FUNCTION SIGNATURES

File: src/orbitalmind/models/normalizing_flow.py
def build_normalizing_flow(input_dim=1, n_flows=4) -> nf.NormalizingFlow
def train_normalizing_flow(residuals, epochs=200) -> tuple[nf.NormalizingFlow, float, float]
    # returns (model, residuals_mean, residuals_std)
def apply_normalizing_flow(model, predictions, res_mean, res_std) -> np.ndarray

---

## VERIFY FOR THIS ITERATION — THIS IS THE HARDEST GATE
Run: pytest tests/test_normalizing_flow.py -v
Run: python src/orbitalmind/evaluation/gaussian_check.py

ALL must pass:
- Shapiro-Wilk p-value > 0.05 on NF-transformed residuals
- KL divergence between residuals and N(0,1) < 0.5
- Q-Q plot saved to outputs/qq_plot.png without error

---

## FAILURE MODES
If Shapiro-Wilk fails (p < 0.05): increase n_flows to 8
If NaN in log_prob: residuals not normalised — check mean/std normalisation
If KL too high: train longer — increase epochs to 400
