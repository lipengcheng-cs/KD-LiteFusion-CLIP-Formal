#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON="/home/lpc/.conda/envs/kdclip/bin/python"
exec "$PYTHON" "$ROOT/scripts/summarize_logits_kd_tuning.py" --root "$ROOT/outputs/logits_kd_tuning"
