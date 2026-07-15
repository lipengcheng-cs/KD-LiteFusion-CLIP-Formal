#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate kdclip
fi

TEACHER_PROJECT_DIR="${TEACHER_PROJECT_DIR:-/home/lpc/projects/KD-LiteFusion-CLIP/mkan_refine/-MKAN-Refine-main}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
TEACHER_CONFIG="${TEACHER_CONFIG:-}"

python3 scripts/export_teacher_logits.py \
  --teacher_project_dir "$TEACHER_PROJECT_DIR" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --teacher_config "$TEACHER_CONFIG" \
  --csv_path data/clean/task2_clean_consistent.csv \
  --image_root data/CrisisMMD_v2.0 \
  --output teacher_cache/mkan_train_logits.pt \
  --expected_samples 6090 \
  2>&1 | tee logs/export_teacher_logits.log

