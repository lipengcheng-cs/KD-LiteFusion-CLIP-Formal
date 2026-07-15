#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/full_wo_kd
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate kdclip
fi

python3 train.py --config configs/full_wo_kd.yaml 2>&1 | tee logs/full_wo_kd.log
