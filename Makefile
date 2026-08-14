# OrbitalMind — Makefile
# Claude Code calls these targets. Never run manually mid-iteration.
# Usage: make <target>

PYTHON = python3
PYTEST = python3 -m pytest
DATA   = data/synthetic/gnss_synthetic.csv

# ── SETUP ───────────────────────────────────────────────

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p models/saved outputs data/synthetic data/raw
	@echo "Setup complete. Run: source venv/bin/activate"

# ── ITERATION VERIFY TARGETS ────────────────────────────
# Each target: runs tests, on pass fires connector automatically

verify-1:
	@echo "[make] Verifying Iteration 1: Synthetic Data"
	$(PYTEST) tests/test_synthetic_data.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 1 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 1 --status fail; \
		exit 1; \
	fi

verify-2:
	@echo "[make] Verifying Iteration 2: Preprocessing"
	$(PYTEST) tests/test_preprocessing.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 2 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 2 --status fail; \
		exit 1; \
	fi

verify-3:
	@echo "[make] Verifying Iteration 3: Features"
	$(PYTEST) tests/test_features.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 3 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 3 --status fail; \
		exit 1; \
	fi

verify-4:
	@echo "[make] Verifying Iteration 4: LSTM + TCN-LSTM + Neural ODE"
	$(PYTEST) tests/test_lstm.py tests/test_neural_ode.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 4 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 4 --status fail; \
		exit 1; \
	fi

verify-5:
	@echo "[make] Verifying Iteration 5: TFT"
	$(PYTEST) tests/test_tft.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 5 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 5 --status fail; \
		exit 1; \
	fi

verify-6:
	@echo "[make] Verifying Iteration 6: Meta-Learner"
	$(PYTEST) tests/test_meta_learner.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 6 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 6 --status fail; \
		exit 1; \
	fi

verify-7:
	@echo "[make] Verifying Iteration 7: Normalizing Flow"
	$(PYTEST) tests/test_normalizing_flow.py -v
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 7 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 7 --status fail; \
		exit 1; \
	fi

verify-8:
	@echo "[make] Verifying Iteration 8: Full Pipeline"
	$(PYTEST) tests/test_pipeline.py -v
	$(PYTHON) src/orbitalmind/run_pipeline.py --data $(DATA) --output outputs
	@if [ $$? -eq 0 ]; then \
		$(PYTHON) scripts/connector.py --iteration 8 --status pass; \
	else \
		$(PYTHON) scripts/connector.py --iteration 8 --status fail; \
		exit 1; \
	fi

# ── FULL REGRESSION ─────────────────────────────────────

test-all:
	@echo "[make] Running full test suite"
	$(PYTEST) tests/ -v --tb=short
	@echo "[make] Done"

# ── CHECKER TRIGGER ─────────────────────────────────────
# Call this after verify passes to request checker review

request-review:
	@echo "[make] Requesting checker review"
	@echo "Open a NEW Claude Code session"
	@echo "Load: .claude/checker/CHECKER.md"
	@echo "Say: Review iteration $$(grep 'Number:' memory/current_iteration.md | head -1 | awk '{print $$2}') using review_code.md and review_tests.md"
	@cat memory/current_iteration.md | grep "Number:"

# ── FORMAT ──────────────────────────────────────────────

format:
	black src/ tests/ scripts/
	@echo "[make] Formatting complete"

format-check:
	black --check src/ tests/ scripts/
	@echo "[make] Format check complete"

# ── CLEAN ────────────────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "[make] Clean complete"

clean-outputs:
	rm -rf outputs/
	mkdir -p outputs/
	@echo "[make] Outputs cleared"

# ── HELP ─────────────────────────────────────────────────

help:
	@echo "OrbitalMind Makefile"
	@echo ""
	@echo "Setup:        make setup"
	@echo "Verify iter:  make verify-1  (1 through 8)"
	@echo "All tests:    make test-all"
	@echo "Format:       make format"
	@echo "Checker:      make request-review"
	@echo "Clean:        make clean"

.PHONY: setup verify-1 verify-2 verify-3 verify-4 verify-5 \
        verify-6 verify-7 verify-8 test-all request-review \
        format format-check clean clean-outputs help

# ── RESUME ───────────────────────────────────────────────

resume:
	@python3 scripts/resume.py

checkpoint:
	@echo "Usage: make checkpoint ITER=1 FN=function_name STATUS=complete NOTE='description'"
	@python3 scripts/write_checkpoint.py \
		--iteration $(ITER) \
		--function "$(FN)" \
		--status $(STATUS) \
		--note "$(NOTE)"
