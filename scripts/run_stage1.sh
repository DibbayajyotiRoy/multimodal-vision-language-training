#!/usr/bin/env bash
set -euo pipefail

# run_stage1.sh - Launch stage 1 training (frozen vision encoder)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${PROJECT_DIR}/configs/train_stage1.yaml}"

cd "${PROJECT_DIR}"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.train --config "${CONFIG_FILE}" --stage stage1
