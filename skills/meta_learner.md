# Skill: LightGBM Meta-Learner
# Iteration 6

---

## GOAL
Fuse outputs from LSTM, TCN-LSTM, TFT, and Neural ODE into one
optimal prediction using LightGBM as the stacking layer.
Final RMSE must improve over best single model.

---

## HOW STACKING WORKS HERE
Each base model produces 96 predictions for the validation period.
Stack them as columns:
    X_meta = [lstm_preds, tcn_preds, tft_preds, node_preds]
    Shape: (n_val_samples, 4)
    y_meta = actual validation values

Train LightGBM on X_meta → y_meta.
At test time, get base model predictions for Day 8, stack same way,
predict with LightGBM → final prediction.

---

## LIGHTGBM CONFIG
objective        = regression
metric           = rmse
num_leaves       = 15       # small — only 4 input features
learning_rate    = 0.05
feature_fraction = 0.9
bagging_fraction = 0.8
bagging_freq     = 5
num_boost_round  = 200
early_stopping   = 30 rounds on validation

---

## FUNCTION SIGNATURES

File: src/orbitalmind/ensemble/lightgbm_meta.py
def train_meta_learner(
    model_outputs: dict,   # {model_name: predictions_array}
    y_true: np.ndarray
) -> lgb.Booster

def predict_meta_learner(
    model: lgb.Booster,
    model_outputs: dict
) -> np.ndarray

def get_feature_importance(model: lgb.Booster, feature_names: list) -> dict

---

## CONSTRAINTS
- model_outputs dict must always have all 4 keys:
  lstm, tcn_lstm, tft, neural_ode
- If any model failed, use zeros array as placeholder — never skip a key
- Save model to models/saved/meta_learner_GEO_clock.txt etc
- Print feature importance after training — tells team which model matters most

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_meta_learner.py -v
ALL must pass:
- Trains without error
- Output shape == (96,)
- RMSE at 1hr < best individual model RMSE at 1hr
- RMSE at 24hr < best individual model RMSE at 24hr
- Feature importance dict has 4 keys with non-zero values

---

## FAILURE MODES
If RMSE worse than single model: val set too small — use all 7 days with CV
If all importance on one model: other models not trained well — check base models
