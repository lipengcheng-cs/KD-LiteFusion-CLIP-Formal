#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/full_logits_kd
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate kdclip
fi

STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-outputs/full_logits_kd/best_weighted_f1.pt}"
python3 evaluate.py \
  --config configs/full_logits_kd.yaml \
  --csv_path data/clean/task2_clean_consistent.csv \
  --image_root data/CrisisMMD_v2.0 \
  --checkpoint "$STUDENT_CHECKPOINT" \
  --output_csv outputs/full_logits_kd/test_predictions.csv \
  --metrics_json outputs/full_logits_kd/eval_metrics.json \
  --per_class_csv outputs/full_logits_kd/per_class_metrics.csv \
  --confusion_csv outputs/full_logits_kd/confusion_matrix.csv \
  2>&1 | tee logs/evaluate_full_logits_kd.log
