.PHONY: help install lint format typecheck test test-fast test-gpu docker-build docker-stage1 docker-stage2 docker-eval clean

help:
	@echo "Available targets:"
	@echo "  install        - Install Python dependencies"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with ruff"
	@echo "  typecheck      - Run mypy type checker"
	@echo "  test           - Run full pytest suite"
	@echo "  test-fast      - Run non-slow tests"
	@echo "  test-gpu       - Run GPU tests"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-stage1  - Run stage 1 training in Docker"
	@echo "  docker-stage2  - Run stage 2 training in Docker"
	@echo "  docker-eval    - Run evaluation in Docker"
	@echo "  clean          - Remove generated artifacts"

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest -v

test-fast:
	pytest -v -m "not slow and not gpu and not requires_model"

test-gpu:
	pytest -v -m gpu

docker-build:
	docker compose build

docker-stage1:
	docker compose up train-stage1

docker-stage2:
	docker compose up train-stage2

docker-eval:
	docker compose up eval

clean:
	rm -rf outputs/ logs/ runs/ checkpoints/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
