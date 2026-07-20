#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI
QUEUE_LOG="$ROOT/logs/mkan_paper_protocol_v2/seed3407_queue.log"
mkdir -p "$(dirname "$QUEUE_LOG")"
cd "$ROOT"

while tmux has-session -t efficiency_diagnosis_20260720 2>/dev/null; do
  echo "WAIT_EFFICIENCY $(date '+%F %T')" | tee -a "$QUEUE_LOG"
  sleep 60
done

idle_samples=0
while (( idle_samples < 3 )); do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')
  root_free_kb=$(df -Pk / | awk 'NR==2 {print $4}')
  shm_free_kb=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
  if (( root_free_kb < 5242880 )); then
    echo "STOP_ROOT_SPACE $(date '+%F %T') root_free_kb=$root_free_kb" | tee -a "$QUEUE_LOG"
    exit 2
  fi
  if (( shm_free_kb < 8388608 )); then
    echo "STOP_SHM_SPACE $(date '+%F %T') shm_free_kb=$shm_free_kb" | tee -a "$QUEUE_LOG"
    exit 3
  fi
  if (( util <= 10 && apps == 0 )); then
    idle_samples=$((idle_samples + 1))
    echo "IDLE_SAMPLE $(date '+%F %T') count=$idle_samples util=$util apps=$apps" | tee -a "$QUEUE_LOG"
  else
    idle_samples=0
    echo "WAIT_GPU $(date '+%F %T') util=$util apps=$apps" | tee -a "$QUEUE_LOG"
  fi
  sleep 30
done

echo "START_BASELINE $(date '+%F %T')" | tee -a "$QUEUE_LOG"
bash scripts/40_run_mkan_paper_seed3407_baseline.sh
echo "BASELINE_PIPELINE_EXITED $(date '+%F %T')" | tee -a "$QUEUE_LOG"

