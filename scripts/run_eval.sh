#!/usr/bin/env bash
set -euo pipefail

# run_eval.sh - Evaluate a trained VLM checkpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-${PROJECT_DIR}/outputs/stage2/best_model.pt}"
CONFIG_FILE="${2:-${PROJECT_DIR}/configs/eval_config.yaml}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.evaluation "${MODEL_PATH}" --config "${CONFIG_FILE}"
