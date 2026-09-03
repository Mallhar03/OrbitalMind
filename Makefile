# OrbitalMind — common tasks
# Run `make help` for the list.

PYTHON  = venv/bin/python3
PYTEST  = venv/bin/python3 -m pytest
REAL    = data/raw/gnss_real.csv
SYNTH   = data/synthetic/gnss_synthetic.csv
WORKERS = 8

# Pin BLAS/OpenMP to one thread per worker. Without this each of the 8 workers
# spawns its own thread pool onto 16 cores and the suite takes 8x longer -- the
# same oversubscription that makes 15 pipeline workers slower than 8.
THREADS = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

.PHONY: setup synthetic fetch run run-fast explain ablation test test-fast \
        format format-check clean clean-outputs help

# ── SETUP ────────────────────────────────────────────────

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p models/saved outputs data/synthetic data/raw
	@echo "Setup complete. Run: source venv/bin/activate, then: make synthetic"

# ── DATA ─────────────────────────────────────────────────
# Nothing under data/ is committed (see data/README.md) — both targets below
# write locally, on demand.

# Small generated dataset for the install smoke test and the unit tests.
synthetic:
	$(PYTHON) scripts/generate_synthetic_data.py

# Pulls real GNSS products from NASA CDDIS. Needs Earthdata credentials in
# ~/.netrc under urs.earthdata.nasa.gov. Writes 7 days of input plus a
# separate holdout file for the 8th day, which training never reads.
fetch:
	$(PYTHON) scripts/fetch_data.py --days 8

# ── PIPELINE ─────────────────────────────────────────────

run:
	$(THREADS) $(PYTHON) src/orbitalmind/run_pipeline.py \
		--data $(REAL) --output outputs --workers $(WORKERS)

# Submission only, skipping the backtest pass. Roughly half the runtime,
# because each satellite then trains its four models once instead of twice.
run-fast:
	$(THREADS) $(PYTHON) src/orbitalmind/run_pipeline.py \
		--data $(REAL) --output outputs --workers $(WORKERS) --no-backtest

# Walk one satellite through every preprocessing stage with before/after stats.
# Usage: make explain SAT=G01
explain:
	@test -n "$(SAT)" || (echo "Usage: make explain SAT=G01"; exit 1)
	$(PYTHON) src/orbitalmind/run_pipeline.py --data $(REAL) --explain $(SAT)

# Which base models, features and signal components actually earn their place.
# Scored on the backtest window only — never on the holdout.
ablation:
	$(THREADS) $(PYTHON) scripts/ablation.py --satellites 6

# ── TESTS ────────────────────────────────────────────────

test:
	$(THREADS) $(PYTEST) tests/ -n $(WORKERS) --tb=short

# Skips the files that train real models end to end: 126 tests in ~2 min.
SLOW = --ignore=tests/test_pipeline.py --ignore=tests/test_lstm.py \
       --ignore=tests/test_tft.py --ignore=tests/test_neural_ode.py \
       --ignore=tests/test_parallel_determinism.py

test-fast:
	$(THREADS) $(PYTEST) tests/ -n $(WORKERS) --tb=short $(SLOW)

# ── HOUSEKEEPING ─────────────────────────────────────────

format:
	black src/ tests/ scripts/

format-check:
	black --check src/ tests/ scripts/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-outputs:
	rm -rf outputs/ && mkdir -p outputs/

help:
	@echo "OrbitalMind"
	@echo ""
	@echo "  make setup       create venv and install dependencies"
	@echo "  make synthetic   generate the local synthetic dataset (smoke test, unit tests)"
	@echo "  make fetch       download 8 days of real GNSS data from NASA CDDIS"
	@echo "  make run         full pipeline with backtest (~3-4.5 h, 95 satellites)"
	@echo "  make run-fast    submission only, no backtest (~half the time)"
	@echo "  make explain SAT=G01   trace one satellite through preprocessing"
	@echo "  make ablation    measure what each model and feature contributes"
	@echo "  make test        full test suite"
	@echo "  make test-fast   test suite without the slow training tests"
	@echo "  make format      black over src/ tests/ scripts/"
	@echo "  make clean       remove bytecode caches"
