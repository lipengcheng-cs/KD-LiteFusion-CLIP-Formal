#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON_BIN="/home/lpc/miniconda3/envs/kdclip/bin/python"
TEACHER_DATA_DIR="/home/lpc/projects/KD-LiteFusion-CLIP/mkan_refine/data"
STUDENT_CSV="${PROJECT_ROOT}/data/clean/task2_clean_consistent.csv"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/formal_kd_stage"
LOG_DIR="${PROJECT_ROOT}/logs/formal_kd_stage"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/audit_teacher_student_data_contract.py \
  --teacher-data-dir "${TEACHER_DATA_DIR}" \
  --student-csv "${STUDENT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/10_audit_data_contract.log"
