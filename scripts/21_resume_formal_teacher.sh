#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
REPRO_ROOT="${PROJECT_ROOT}/mkan_refine/reproduction"
PYTHON_BIN="/home/lpc/miniconda3/envs/kdclip/bin/python"
CONFIG="${REPRO_ROOT}/configs/kd_formal_teacher.yaml"
LOG_ROOT="${PROJECT_ROOT}/logs/formal_kd_stage"

# The shared root filesystem previously filled while multiprocessing workers
# attempted to create /tmp resources. Training now uses num_workers=0, and any
# remaining temporary files are directed to memory-backed storage.
export TMPDIR="/dev/shm/lpc_kdclip_tmp"
mkdir -p "${TMPDIR}" "${LOG_ROOT}"

cd "${REPRO_ROOT}"
"${PYTHON_BIN}" train_formal_teacher.py --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/22_resume_formal_teacher.log"

cd "${PROJECT_ROOT}"
bash scripts/11_optimize_teacher_ensemble.sh
bash scripts/12_export_and_check_formal_teacher_cache.sh
