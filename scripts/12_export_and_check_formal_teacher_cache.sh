#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
REPRO_ROOT="${PROJECT_ROOT}/mkan_refine/reproduction"
PYTHON_BIN="/home/lpc/miniconda3/envs/kdclip/bin/python"
CONFIG="${REPRO_ROOT}/configs/kd_formal_teacher.yaml"
TEACHER_ROOT="${PROJECT_ROOT}/outputs/server_mkan_kd_formal"
CSV_PATH="${PROJECT_ROOT}/data/clean/task2_clean_consistent.csv"
LOG_DIR="${PROJECT_ROOT}/logs/formal_kd_stage"

mkdir -p "${TEACHER_ROOT}/teacher_cache" "${LOG_DIR}"
test -s "${TEACHER_ROOT}/reports/ensemble_selected_weights.json"
cd "${REPRO_ROOT}"

"${PYTHON_BIN}" export_formal_teacher_cache.py --config "${CONFIG}" \
  2>&1 | tee "${LOG_DIR}/23_export_formal_teacher_cache.log"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/check_formal_teacher_cache.py \
  --csv "${CSV_PATH}" \
  --logits-cache "${TEACHER_ROOT}/teacher_cache/mkan_train_logits.pt" \
  --full-cache "${TEACHER_ROOT}/teacher_cache/mkan_train_full.pt" \
  --report "${TEACHER_ROOT}/teacher_cache/check_report.json" \
  2>&1 | tee "${LOG_DIR}/24_check_formal_teacher_cache.log"
