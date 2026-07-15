#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
REPRO_ROOT="${PROJECT_ROOT}/mkan_refine/reproduction"
PYTHON_BIN="/home/lpc/miniconda3/envs/kdclip/bin/python"
CONFIG="${REPRO_ROOT}/configs/kd_formal_teacher.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/server_mkan_kd_formal"
LOG_ROOT="${PROJECT_ROOT}/logs/formal_kd_stage"

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/teacher_cache" \
  "${OUTPUT_ROOT}/reports" "${LOG_ROOT}"
cd "${REPRO_ROOT}"

"${PYTHON_BIN}" precompute_formal_clip_features.py --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/20_precompute_formal_clip_features.log"

"${PYTHON_BIN}" train_formal_teacher.py --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/21_train_formal_teacher.log"

cd "${PROJECT_ROOT}"
bash scripts/11_optimize_teacher_ensemble.sh
bash scripts/12_export_and_check_formal_teacher_cache.sh
