#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI
CACHE=/dev/shm/lpc_kdclip_tmp/mkan_paper_protocol_v2
CONFIG="$ROOT/mkan_refine/paper_reproduction_v2/configs/paper_protocol.yaml"
LOG_DIR="$ROOT/logs/mkan_paper_protocol_v2"
mkdir -p "$CACHE" "$LOG_DIR"

source /opt/miniconda3/bin/activate
conda activate kdclip
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

df -h /
df -h /dev/shm
nvidia-smi

python -m mkan_refine.paper_reproduction_v2.precompute_features \
  --data-dir "$ROOT/mkan_refine/data" \
  --image-root "$ROOT/data/CrisisMMD_v2.0" \
  --cache-root "$CACHE" \
  --clip-checkpoint /home/lpc/.cache/clip/ViT-L-14-336px.pt \
  --splits train val \
  --batch-size 16 2>&1 | tee -a "$LOG_DIR/seed3407_baseline.log"

python -m mkan_refine.paper_reproduction_v2.train \
  --config "$CONFIG" \
  --seed 3407 2>&1 | tee -a "$LOG_DIR/seed3407_baseline.log"

