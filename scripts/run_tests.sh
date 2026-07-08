#!/usr/bin/env bash
set -euo pipefail

# run_tests.sh - Run linting, type checks, and tests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "Running ruff linter"
ruff check src tests scripts

echo "Running mypy"
mypy src

echo "Running pytest"
pytest -v -m "not slow and not gpu and not requires_model" "$@"
