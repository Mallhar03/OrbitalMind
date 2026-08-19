# Skill: Normalizing Flow Post-Processor
# Iteration 7

---

## GOAL
Learn the distribution of the ensemble's out-of-sample residuals, then use it
to (a) bias-correct the point forecast and (b) attach a predictive
distribution to every forecast point.

## READ THIS FIRST — THE TRAP THIS SKILL ALREADY CAUSED ONCE
An earlier wording of this file said "make residuals Gaussian by construction"
and "residuals follow N(0, sigma^2) when evaluated by judges". That is an
instruction to manufacture the metric, and the implementation did exactly
that. It computed

    _correction = (z_gaussian * res_std + res_mean) - residuals
    corrected   = preds - _correction

from the validation ground truth, which made (actual - corrected) a vector of
exact normal quantiles BY DEFINITION. Shapiro-Wilk then returned p = 0.9999
for every possible input, including residuals that were half constant and half
exponential. It measured nothing, and the same ground-truth-derived vector was
subtracted from the day-8 forecast, degrading it.

Gaussian residuals are a property the model must EARN, not a number the
post-processor writes down. If the residuals are not Gaussian, the pipeline
must say so. See memory/what_failed.md, iteration 9.

---

## WHAT THIS MODULE DOES
The calibration window is split in half by orbitalmind.splits. The
meta-learner fits on the first half; the flow fits on the second half, so the
residuals it sees are out-of-sample for the base models AND the meta-learner.

    residuals = actual - predicted        (on the flow's half of calibration)
    fit the flow to those residuals by maximum likelihood
    draw a sample pool from the fitted flow, map back to residual units
    bias   = median(pool)      -> a SCALAR shift applied to the forecast
    bounds = quantiles(pool)   -> the predictive interval

The shift is a single scalar. It cannot encode anything index-specific about
the window being predicted, which is the property that makes it honest and
which tests/test_normalizing_flow.py asserts directly.

Because reconstruction sums differenced steps, the pipeline widens the
interval as sqrt(k) across the horizon.

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
def train_normalizing_flow(residuals, epochs=200, orbit_type='GEO',
                           error_col='ClockError_ns') -> ResidualCalibration
def apply_normalizing_flow(calibration, predictions) -> np.ndarray
    # bias-corrected point forecast; the shift is a scalar
def predictive_interval(calibration, predictions, level=0.95) -> (lower, upper, sigma)
def shapiro_wilk(residuals) -> (statistic, p_value, 'PASS'|'FAIL')
    # reports what the residuals actually are

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_normalizing_flow.py -v

ALL must pass:
- the correction is a scalar shift, identical for any two prediction vectors
- deliberately non-Gaussian residuals do NOT come back with a high p-value
- the 95% interval covers roughly 95% of fresh residuals — not 100%
- interval width grows with residual spread
- Q-Q plot saved to outputs/qq_plot.png without error

The pipeline's reported Shapiro-Wilk p-value is an OUTCOME, not a gate. A FAIL
is a real result that tells you the ensemble's errors are structured and there
is signal left to model. Do not tune the post-processor until it passes.

---

## FAILURE MODES
If NaN/inf in log_prob: the flow diverged. train_normalizing_flow already
    falls back to a Gaussian fitted to the empirical mean and std, and sets
    calibration.fitted = False. Check that residuals were standardised.
If the calibration residuals are constant: sigma is 0 and the interval
    collapses. That means the base models are predicting a constant — look at
    the base models, not at the flow.
If Shapiro-Wilk FAILS on held-out residuals: that is information, not a bug.
    The residuals carry structure the ensemble has not captured. Fix it in the
    models or the features. NEVER fix it in the post-processor by reaching for
    the ground truth — that is the exact defect logged in
    memory/what_failed.md, iteration 9.
