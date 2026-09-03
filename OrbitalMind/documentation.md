# OrbitalMind Setup & Troubleshooting Documentation

This document keeps track of environment setup steps and troubleshooting fixes applied to the OrbitalMind pipeline.

## 1. IDE Module Resolution (`Cannot find module 'PyEMD'`)
**Symptom:** Your editor (e.g., VS Code) highlights `from PyEMD import EMD` with an error, claiming the module cannot be found, even though `requirements.txt` has been installed.
**Cause:** The IDE is using the global Python interpreter (e.g., `AppData\Local\Programs\Python\Python311`) rather than the project's virtual environment (`venv`). Note that the package name on PyPI is `EMD-signal`, not `PyEMD` (running `pip install PyEMD` installs the wrong package).
**Fix:** 
1. Press `Ctrl+Shift+P` in VS Code.
2. Select **Python: Select Interpreter**.
3. Choose `d:\OrbitalMind\venv\Scripts\python.exe`.

## 2. Synthetic Data Generation Missing
**Symptom:** Running the smoke test (`python src/orbitalmind/run_pipeline.py --data data/synthetic/gnss_synthetic.csv --max-satellites 2`) fails with a `FileNotFoundError` because `gnss_synthetic.csv` does not exist.
**Cause:** The synthetic data is not tracked in git and needs to be generated locally, but simply running `synthetic_generator.py` does not execute anything because it lacked a `__main__` block.
**Fix:** The generator was executed by importing it directly via the command line:
```powershell
$env:PYTHONPATH="src"; venv\Scripts\python.exe -c "from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data; generate_synthetic_gnss_data()"
```

## 3. Synthetic Data Column Mismatch
**Symptom:** Running the pipeline against the synthetic data throws `ValueError: input CSV missing required columns: ['satclockerror (m)', 'x_error (m)', 'y_error (m)', 'z_error (m)']`.
**Cause:** The generator (`synthetic_generator.py`) was outdated and produced columns named `ClockError_ns` and `EphemerisError_m`, while `run_pipeline.py` strictly expects all four real-world component columns.
**Fix:** `synthetic_generator.py` was updated to explicitly write `satclockerror (m)`, `x_error (m)`, `y_error (m)`, and `z_error (m)`. The dataset was regenerated and successfully allowed the pipeline to run.

## 4. Backtest Dimensionality Mismatch Warning
**Symptom:** During the pipeline run, warnings like `operands could not be broadcast together with shapes (95,) (96,)` are printed for the backtest window.
**Status:** This is a minor misalignment when the 96-step forecast is evaluated against a 95-row slice of actual holdout data. It forces the backtest to fall back to the persistence baseline for scoring on those satellites. The pipeline still completes and successfully generates `submission.csv`.
