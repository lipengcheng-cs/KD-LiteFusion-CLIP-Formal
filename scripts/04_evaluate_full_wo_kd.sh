#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/full_wo_kd
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate kdclip
fi

python3 evaluate.py \
  --config configs/full_wo_kd.yaml \
  --csv_path data/clean/task2_clean_consistent.csv \
  --image_root data/CrisisMMD_v2.0 \
  --checkpoint outputs/full_wo_kd/best.pt \
  --output_csv outputs/full_wo_kd/test_predictions.csv \
  --metrics_json outputs/full_wo_kd/eval_metrics.json \
  --per_class_csv outputs/full_wo_kd/per_class_metrics.csv \
  --confusion_csv outputs/full_wo_kd/confusion_matrix.csv \
  2>&1 | tee logs/evaluate_full_wo_kd.log
