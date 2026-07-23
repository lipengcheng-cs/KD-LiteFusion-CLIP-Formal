#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
ROOT="outputs/litefusion_v2/formal_wo_kd_multiseed"
LOG_DIR="logs/litefusion_v2/student_training/formal_wo_kd_multiseed"
STATUS="${LOG_DIR}/pipeline_status.log"

source /opt/miniconda3/bin/activate
conda activate kdclip
cd "${PROJECT_DIR}"
export TMPDIR="/dev/shm/lpc_kdclip_tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}" "${ROOT}" "${LOG_DIR}"

reuse_seed3407() {
    local source_dir="$1"
    local target_dir="$2"
    if [[ -f "${target_dir}/COMPLETED" ]]; then
        return
    fi
    if [[ -e "${target_dir}" ]]; then
        echo "Refusing non-empty/reused target: ${target_dir}" >> "${STATUS}"
        exit 1
    fi
    if [[ ! -f "${source_dir}/COMPLETED" ]]; then
        echo "Missing completed seed3407 source: ${source_dir}" >> "${STATUS}"
        exit 1
    fi
    cp -a "${source_dir}" "${target_dir}"
    printf 'Reused identical final-config seed3407 run from %s; training was not duplicated.\n' \
        "${source_dir}" > "${target_dir}/REUSED_FROM.txt"
}

run_one() {
    local candidate="$1"
    local config="$2"
    local seed="$3"
    local output_dir="${ROOT}/${candidate}/seed${seed}"
    local log_path="${LOG_DIR}/${candidate}_seed${seed}.log"
    if [[ -f "${output_dir}/COMPLETED" ]]; then
        printf '%s SKIP %s seed%s\n' "$(date --iso-8601=seconds)" "${candidate}" "${seed}" >> "${STATUS}"
        return
    fi
    if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf '%s ERROR non-empty %s\n' "$(date --iso-8601=seconds)" "${output_dir}" >> "${STATUS}"
        exit 1
    fi
    printf '%s START %s seed%s\n' "$(date --iso-8601=seconds)" "${candidate}" "${seed}" >> "${STATUS}"
    python scripts/run_litefusion_v2_student_training.py \
        --config "configs/litefusion_v2/${config}.yaml" \
        --config-name "${candidate}_formal_wo_kd_seed${seed}" \
        --output-dir "${output_dir}" \
        --seed "${seed}" --epochs 10 --batch-size 8 --num-workers 0 \
        --lr 2e-4 --weight-decay 0.01 --scheduler none --warmup-epochs 0 \
        --min-lr 1e-6 --class-weight-method inverse_freq \
        --label-smoothing 0.05 --dropout 0.2 --max-grad-norm 1.0 \
        > "${log_path}" 2>&1
    printf '%s COMPLETE %s seed%s\n' "$(date --iso-8601=seconds)" "${candidate}" "${seed}" >> "${STATUS}"
}

mkdir -p "${ROOT}/v2_c_compact" "${ROOT}/v2_g_grouped"
reuse_seed3407 \
    "outputs/litefusion_v2/student_optimization/c_compact_seed3407_baseline_10ep" \
    "${ROOT}/v2_c_compact/seed3407"
reuse_seed3407 \
    "outputs/litefusion_v2/student_optimization/g_grouped_optimized_seed3407_10ep" \
    "${ROOT}/v2_g_grouped/seed3407"

run_one v2_c_compact v2_c_compact 42
run_one v2_c_compact v2_c_compact 2024
run_one v2_g_grouped v2_g_grouped 42
run_one v2_g_grouped v2_g_grouped 2024
printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)" >> "${STATUS}"
