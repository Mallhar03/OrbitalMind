# OrbitalMind
### Physics-Aware Ensemble AI for GNSS Clock & Ephemeris Error Prediction
**Team Xenith | SH-DST-03 | Smart Horizon 2026**

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python src/orbitalmind/run_pipeline.py --data data/synthetic/gnss_synthetic.csv

# 4. Run tests
pytest tests/ -v
```

---

## Build Status

| Iteration | Module | Status |
|-----------|--------|--------|
| 1 | Synthetic Data Generator | ✅ Complete |
| 2 | Preprocessing Pipeline | ✅ Complete |
| 3 | Feature Engineering | ✅ Complete |
| 4 | LSTM + TCN-LSTM + Neural ODE | ✅ Complete |
| 5 | TFT Model | ✅ Complete |
| 6 | LightGBM Meta-Learner | ⬜ Not started |
| 7 | Normalizing Flow | ⬜ Not started |
| 8 | Full Pipeline Integration | ⬜ Not started |

---

## Verify Conditions (v1 Pass)
- RMSE at 1hr < 2.0 ns on synthetic data
- Shapiro-Wilk p > 0.05 on final residuals
- Full pipeline runs in one command under 30 minutes
