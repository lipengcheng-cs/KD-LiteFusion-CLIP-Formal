#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON_BIN="/home/lpc/miniconda3/envs/kdclip/bin/python"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/formal_multiseed"
LOG_DIR="${PROJECT_ROOT}/logs/formal_multiseed"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_matched_multiseed_experiments.py \
  --project-root "${PROJECT_ROOT}" \
  --python "${PYTHON_BIN}" \
  2>&1 | tee "${LOG_DIR}/12_run_matched_multiseed.log"

"${PYTHON_BIN}" scripts/summarize_multiseed_results.py \
  --root "${OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/12_summarize_multiseed.log"
