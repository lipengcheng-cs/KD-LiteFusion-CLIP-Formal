#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

main() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate kdclip
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" != "kdclip" ]]; then
    echo "ERROR: expected conda environment kdclip, got ${CONDA_DEFAULT_ENV:-NONE}" >&2
    return 1
  fi

  CLIP_PATH=$(python3 - <<'PY'
import yaml
with open("configs/full_wo_kd.yaml", "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["model"]["clip_model_path"])
PY
)
  echo "current_dir: $(pwd)"
  echo "python: $(command -v python3)"
  echo "conda_env: ${CONDA_DEFAULT_ENV}"

python3 - <<'PY'
import torch

print("pytorch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

  if [[ ! -f "$CLIP_PATH" ]]; then
    echo "ERROR: local OpenAI CLIP checkpoint not found: $CLIP_PATH" >&2
    return 1
  fi
  export CLIP_PATH

python3 - <<'PY'
import os
import clip

path = os.environ["CLIP_PATH"]
models = clip.available_models()
print("clip_available_models:", models)
if "ViT-L/14@336px" not in models:
    raise RuntimeError("OpenAI CLIP does not report ViT-L/14@336px as available")
size = os.path.getsize(path)
print("clip_checkpoint_bytes:", size)
if size <= 800 * 1024 * 1024:
    raise RuntimeError(f"CLIP checkpoint is unexpectedly small: {size} bytes")
clip.load(path, device="cpu", jit=False)
print("local OpenAI CLIP load: OK")
PY
}

main 2>&1 | tee logs/check_env.log
