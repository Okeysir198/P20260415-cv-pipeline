.PHONY: lint format format-check test test-core test-tools test-utils train evaluate error-analysis export demo health test-dataset test-pipeline

# Linting
lint:
	uv run ruff check core/ utils/ tests/ scripts/

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

# Testing
test:
	uv run pytest tests/ -v --tb=short

test-core:
	uv run pytest tests/test_core*.py -v --tb=short

test-tools:
	uv run pytest tests/test_tools*.py -v --tb=short

test-utils:
	uv run pytest tests/test_utils*.py -v --tb=short

# Pipeline shortcuts (usage: make train CONFIG=configs/fire/06_training.yaml)
train:
	uv run core/p06_training/train.py --config $(CONFIG)

evaluate:
	uv run core/p08_evaluation/evaluate.py --model $(MODEL) --config $(CONFIG)

error-analysis:
	uv run core/p08_evaluation/evaluate.py --model $(MODEL) --config $(CONFIG) --error-analysis --save-dir $(SAVE_DIR)

export:
	uv run core/p09_export/export.py --model $(MODEL) --training-config $(CONFIG)

demo:
	uv run app_demo/run.py

# Service health checks
health:
	@echo "=== Service Health Checks ==="
	@echo -n "SAM3 (:18100): "; curl -sf http://localhost:18100/health && echo "UP" || echo "DOWN"
	@echo -n "Flux NIM (:18101): "; curl -sf http://localhost:18101/v1/health/ready && echo "UP" || echo "DOWN"
	@echo -n "Image Editor (:18102): "; curl -sf http://localhost:18102/health && echo "UP" || echo "DOWN"
	@echo -n "Label Studio (:18103): "; curl -sf http://localhost:18103/health && echo "UP" || echo "DOWN"
	@echo -n "Auto-Label (:18104): "; curl -sf http://localhost:18104/health && echo "UP" || echo "DOWN"
	@echo -n "Annotation QA (:18105): "; curl -sf http://localhost:18105/health && echo "UP" || echo "DOWN"

# Dataset setup
test-dataset:
	uv run tests/create_test_dataset.py

# Full pipeline test (sequential)
test-pipeline:
	uv run tests/run_all.py
