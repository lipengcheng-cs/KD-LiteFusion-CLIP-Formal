#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
OUTPUT_DIR="$PROJECT_ROOT/outputs/efficiency/diagnosis"
LOG_DIR="$PROJECT_ROOT/logs/efficiency"
TMP_ROOT="/dev/shm/lpc_kdclip_tmp"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

df -h /
df -h /dev/shm
nvidia-smi
ps -ef | grep -E "train.py|evaluate.py|benchmark|efficiency.py|screening" | grep -v grep || true

source /opt/miniconda3/bin/activate
conda activate kdclip

python scripts/profile_end_to_end_components.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --clip-model-path /home/lpc/.cache/clip/ViT-L-14-336px.pt \
  --warmup 30 \
  --iterations 100 \
  --rounds 3 \
  --batch-sizes 1 8 \
  2>&1 | tee "$LOG_DIR/diagnose_efficiency_components.log"

python scripts/diagnose_efficiency_anomaly.py \
  --project-root "$PROJECT_ROOT" \
  --diagnosis-dir "$OUTPUT_DIR" \
  --old-raw "$PROJECT_ROOT/outputs/efficiency/raw_benchmark.json" \
  2>&1 | tee "$LOG_DIR/diagnose_efficiency_summary.log"

echo "EFFICIENCY_DIAGNOSIS_COMPLETE $(date -Is)"
