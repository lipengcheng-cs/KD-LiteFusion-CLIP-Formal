#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON="/home/lpc/.conda/envs/kdclip/bin/python"
exec "$PYTHON" "$ROOT/scripts/generate_next_stage_formal_report.py" --project-root "$ROOT"
