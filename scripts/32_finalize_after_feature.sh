#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lpc/projects/KD-LiteFusion-CLIP_FULL_RUN_MINI"
cd "$ROOT"

while tmux has-session -t feature_kd_screen_20260717 2>/dev/null; do
  date
  df -h /
  sleep 60
done

if [[ ! -f outputs/feature_kd_screening/screening_manifest.json ]]; then
  echo "ERROR: Feature KD tmux ended without screening_manifest.json" >&2
  exit 1
fi

scripts/28_summarize_feature_kd.sh
cp /dev/shm/CODE_CHANGELOG_NEXT_FORMAL_STAGE.md CODE_CHANGELOG_NEXT_FORMAL_STAGE.md
git add CODE_CHANGELOG_NEXT_FORMAL_STAGE.md train.py kd_litefusion_mkan_teacher/teacher_cache.py kd_litefusion_mkan_teacher/losses.py scripts/run_feature_kd_screening.py scripts/27_run_feature_kd_screening.sh scripts/summarize_feature_kd.py scripts/28_summarize_feature_kd.sh configs/feature_kd
git add -f outputs/feature_kd_screening/screening_manifest.json outputs/feature_kd_screening/screening_results.csv outputs/feature_kd_screening/feature_kd_report.md
[[ -f outputs/feature_kd_screening/selected_feature_config.yaml ]] && git add -f outputs/feature_kd_screening/selected_feature_config.yaml
[[ -f outputs/feature_kd_screening/selected_multiseed_test_results.csv ]] && git add -f outputs/feature_kd_screening/selected_multiseed_test_results.csv
git commit -m "Add Feature KD screening" || true

df -h /
df -h /dev/shm
nvidia-smi
scripts/29_analyze_formal_teacher_gate.sh
git add scripts/analyze_formal_teacher_gate.py scripts/29_analyze_formal_teacher_gate.sh
git add -f outputs/gate_analysis
git commit -m "Add formal teacher gate diagnostics" || true

scripts/31_generate_next_stage_formal_report.sh
git add scripts/generate_next_stage_formal_report.py scripts/31_generate_next_stage_formal_report.sh CODE_CHANGELOG_NEXT_FORMAL_STAGE.md
git add -f outputs/NEXT_STAGE_FORMAL_KD_REPORT.md outputs/FORMAL_EXPERIMENT_MATRIX.csv outputs/FORMAL_RESULTS_SUMMARY.json outputs/FORMAL_RESULTS_SUMMARY.md
git commit -m "Add next-stage formal KD report" || true

git status --short
git log -8 --oneline
df -h /
