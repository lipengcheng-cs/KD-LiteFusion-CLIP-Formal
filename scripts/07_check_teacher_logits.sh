#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/cache_check
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate kdclip
fi

python3 scripts/check_teacher_logits.py \
  --cache teacher_cache/mkan_train_logits.pt \
  --csv_path data/clean/task2_clean_consistent.csv \
  --report outputs/cache_check/logits_cache_report.json \
  --expected_samples 6090 \
  2>&1 | tee logs/check_teacher_logits.log

