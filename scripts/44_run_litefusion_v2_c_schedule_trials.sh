#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
LOG_DIR="${PROJECT_DIR}/logs/litefusion_v2/student_training"
STATUS_LOG="${LOG_DIR}/c_schedule_pipeline_status.log"
BASELINE_DIR="outputs/litefusion_v2/student_optimization/c_compact_seed3407_baseline_10ep"

source /opt/miniconda3/bin/activate
conda activate kdclip
cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"
export TMPDIR="/dev/shm/lpc_kdclip_tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

if [[ ! -f "${BASELINE_DIR}/COMPLETED" ]]; then
    printf '%s ERROR missing completed C0 baseline\n' "$(date --iso-8601=seconds)" >> "${STATUS_LOG}"
    exit 1
fi

run_experiment() {
    local experiment_name="$1"
    local output_dir="$2"
    local learning_rate="$3"
    local log_path="$4"

    if [[ -f "${output_dir}/COMPLETED" ]]; then
        printf '%s SKIP completed %s\n' "$(date --iso-8601=seconds)" "${experiment_name}" >> "${STATUS_LOG}"
        return
    fi
    if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf '%s ERROR non-empty output %s\n' "$(date --iso-8601=seconds)" "${output_dir}" >> "${STATUS_LOG}"
        return 1
    fi

    printf '%s START %s\n' "$(date --iso-8601=seconds)" "${experiment_name}" >> "${STATUS_LOG}"
    python scripts/run_litefusion_v2_student_training.py \
        --config configs/litefusion_v2/v2_c_compact.yaml \
        --config-name "${experiment_name}" \
        --output-dir "${output_dir}" \
        --seed 3407 \
        --epochs 10 \
        --batch-size 8 \
        --num-workers 0 \
        --lr "${learning_rate}" \
        --weight-decay 0.01 \
        --scheduler cosine \
        --warmup-epochs 1 \
        --min-lr 1e-6 \
        --class-weight-method inverse_freq \
        --label-smoothing 0.05 \
        --dropout 0.2 \
        --max-grad-norm 1.0 2>&1 | tee "${log_path}"
    printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "${experiment_name}" >> "${STATUS_LOG}"
}

run_experiment \
    c_compact_C1_cosine_lr2e4 \
    outputs/litefusion_v2/student_optimization/c_compact_C1_cosine_lr2e4 \
    2e-4 \
    "${LOG_DIR}/c_compact_C1_cosine_lr2e4.log"

run_experiment \
    c_compact_C2_cosine_lr1e4 \
    outputs/litefusion_v2/student_optimization/c_compact_C2_cosine_lr1e4 \
    1e-4 \
    "${LOG_DIR}/c_compact_C2_cosine_lr1e4.log"

printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)" >> "${STATUS_LOG}"
