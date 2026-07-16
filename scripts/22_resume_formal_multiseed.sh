#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON="/home/lpc/.conda/envs/kdclip/bin/python"

df -h /
df -h /dev/shm
nvidia-smi
pgrep -af "train.py|12_run_matched_multiseed" || true
exec "$PYTHON" "$ROOT/scripts/resume_formal_multiseed.py" --project-root "$ROOT" --python "$PYTHON" "$@"
