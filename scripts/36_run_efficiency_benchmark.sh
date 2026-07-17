#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
PYTHON="/home/lpc/.conda/envs/kdclip/bin/python"
OUT="$ROOT/outputs/efficiency"
LOG="$ROOT/logs/efficiency/formal_efficiency.log"

cd "$ROOT"
mkdir -p "$OUT" "$(dirname "$LOG")" /dev/shm/lpc_kdclip_tmp
export TMPDIR=/dev/shm/lpc_kdclip_tmp

df -h /
df -h /dev/shm
nvidia-smi

ACTIVE_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$ACTIVE_GPU_PIDS" ]]; then
  echo "ERROR: GPU has active compute processes: $ACTIVE_GPU_PIDS" >&2
  exit 2
fi
GPU_UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ "${GPU_UTIL:-100}" -gt 10 ]]; then
  echo "ERROR: GPU utilization is ${GPU_UTIL}%; refusing formal measurement" >&2
  exit 3
fi

exec > >(tee -a "$LOG") 2>&1
echo "FORMAL_EFFICIENCY_START $(date --iso-8601=seconds)"
"$PYTHON" efficiency.py \
  --project-root "$ROOT" \
  --output-dir "$OUT" \
  --clip-model-path /home/lpc/.cache/clip/ViT-L-14-336px.pt \
  --warmup 30 \
  --iterations 100 \
  --rounds 3 \
  --batch-sizes 1 8

"$PYTHON" scripts/summarize_efficiency.py --project-root "$ROOT" --output-dir "$OUT"

required=(
  model_parameter_breakdown.csv flops_macs_report.csv latency_batch1.csv latency_batch8.csv
  throughput.csv gpu_memory.csv checkpoint_sizes.csv relative_reduction_rates.csv
  performance_efficiency_tradeoff.csv efficiency_report.md
  weighted_f1_vs_parameters.png macro_f1_vs_latency.png performance_vs_flops.png
  teacher_student_efficiency_comparison.png
)
for name in "${required[@]}"; do
  [[ -s "$OUT/$name" ]] || { echo "ERROR: missing/empty $OUT/$name" >&2; exit 4; }
done

echo "FORMAL_EFFICIENCY_PASS $(date --iso-8601=seconds)"
