#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON="/home/lpc/.conda/envs/kdclip/bin/python"

df -h /
df -h /dev/shm
nvidia-smi
if pgrep -u "$(id -u)" -af "train.py"; then
  echo "ERROR: an existing train.py process is active; refusing duplicate launch" >&2
  exit 2
fi
mkdir -p /dev/shm/lpc_kdclip_tmp
export TMPDIR=/dev/shm/lpc_kdclip_tmp
exec "$PYTHON" "$ROOT/scripts/run_feature_kd_screening.py" --project-root "$ROOT" --python "$PYTHON"
