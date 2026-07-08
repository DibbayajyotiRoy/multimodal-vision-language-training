#!/usr/bin/env bash
set -euo pipefail

# run_stage2.sh - Launch stage 2 training (tune remaining stage)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${PROJECT_DIR}/configs/train_stage2.yaml}"
RESUME_FROM="${2:-${PROJECT_DIR}/outputs/stage1/best_model.pt}"

cd "${PROJECT_DIR}"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.train \
    --config "${CONFIG_FILE}" \
    --stage stage2 \
    --resume_from "${RESUME_FROM}"
