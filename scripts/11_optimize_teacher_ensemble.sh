#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON_BIN="/home/lpc/miniconda3/envs/kdclip/bin/python"
TEACHER_ROOT="${PROJECT_ROOT}/outputs/server_mkan_kd_formal"
LOG_DIR="${PROJECT_ROOT}/logs/formal_kd_stage"

mkdir -p "${TEACHER_ROOT}/reports" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

for seed in 3407 42 2024; do
  test -s "${TEACHER_ROOT}/seed_${seed}/val_predictions.csv"
  test -s "${TEACHER_ROOT}/seed_${seed}/test_predictions.csv"
done

"${PYTHON_BIN}" scripts/optimize_teacher_ensemble.py \
  --teacher-root "${TEACHER_ROOT}" \
  2>&1 | tee "${LOG_DIR}/22_optimize_teacher_ensemble.log"
