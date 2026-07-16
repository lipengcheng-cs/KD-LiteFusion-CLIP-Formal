#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON="/home/lpc/.conda/envs/kdclip/bin/python"
"$PYTHON" "$ROOT/scripts/check_formal_multiseed_completion.py" --project-root "$ROOT"
exec "$PYTHON" "$ROOT/scripts/summarize_formal_multiseed.py" --root "$ROOT/outputs/formal_multiseed"
