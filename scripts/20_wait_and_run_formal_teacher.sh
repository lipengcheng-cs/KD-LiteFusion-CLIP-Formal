#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
LOG_DIR="${PROJECT_ROOT}/logs/formal_kd_stage"
WAIT_LOG="${LOG_DIR}/20_wait_for_gpu.log"
MIN_FREE_MIB=12000
MAX_UTIL_PERCENT=20
REQUIRED_STABLE_CHECKS=3

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

stable=0
while true; do
  IFS=',' read -r free_mib utilization <<< "$(
    nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits |
      head -n 1 | tr -d ' '
  )"
  timestamp="$(date --iso-8601=seconds)"
  printf '%s free_mib=%s utilization=%s stable_checks=%s/%s\n' \
    "${timestamp}" "${free_mib}" "${utilization}" "${stable}" "${REQUIRED_STABLE_CHECKS}" |
    tee -a "${WAIT_LOG}"
  if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTIL_PERCENT )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  if (( stable >= REQUIRED_STABLE_CHECKS )); then
    break
  fi
  sleep 60
done

printf '%s GPU safety gate passed; starting formal teacher pipeline.\n' "$(date --iso-8601=seconds)" |
  tee -a "${WAIT_LOG}"
bash scripts/run_formal_teacher_pipeline.sh
