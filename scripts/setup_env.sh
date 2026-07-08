#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh - Bootstrap project environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

cd "${PROJECT_DIR}"

echo "Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Upgrading base packages"
pip install --upgrade pip setuptools wheel

echo "Installing dependencies"
pip install -r requirements.txt

echo "Installing project in editable mode"
pip install -e .

if command -v git &> /dev/null; then
    echo "Installing pre-commit hooks"
    pre-commit install || true
fi

echo "Generating dummy dataset for smoke tests"
python -m src.data --output_dir data/dummy --num_samples 200

echo "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
