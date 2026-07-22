#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
LOG_DIR="${PROJECT_DIR}/logs/litefusion_v2/student_training"
STATUS_LOG="${LOG_DIR}/baseline_pipeline_status.log"

source /opt/miniconda3/bin/activate
conda activate kdclip
cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"
export TMPDIR="/dev/shm/lpc_kdclip_tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

run_experiment() {
    local experiment_name="$1"
    local config_name="$2"
    local output_dir="$3"
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
        --config "configs/litefusion_v2/${config_name}.yaml" \
        --config-name "${experiment_name}" \
        --output-dir "${output_dir}" \
        --seed 3407 \
        --epochs 10 \
        --batch-size 8 \
        --num-workers 0 \
        --lr 2e-4 \
        --weight-decay 0.01 \
        --scheduler none \
        --warmup-epochs 0 \
        --class-weight-method inverse_freq \
        --label-smoothing 0.05 \
        --dropout 0.2 \
        --max-grad-norm 1.0 2>&1 | tee "${log_path}"
    printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "${experiment_name}" >> "${STATUS_LOG}"
}

run_experiment \
    "c_compact_seed3407_baseline_10ep" \
    "v2_c_compact" \
    "outputs/litefusion_v2/student_optimization/c_compact_seed3407_baseline_10ep" \
    "${LOG_DIR}/c_compact_seed3407_baseline_10ep.log"

run_experiment \
    "p_precision_seed3407_baseline_10ep" \
    "v2_p_precision" \
    "outputs/litefusion_v2/student_optimization/p_precision_seed3407_baseline_10ep" \
    "${LOG_DIR}/p_precision_seed3407_baseline_10ep.log"

printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)" >> "${STATUS_LOG}"
