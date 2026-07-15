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

python3 - <<'PY'
import os
import random

import pandas as pd
from PIL import Image

data_root = "data/CrisisMMD_v2.0"
csv_path = "data/clean/task2_clean_consistent.csv"
required = {"sample_id", "image_path", "text", "label", "split"}

if not os.path.isdir(data_root):
    raise FileNotFoundError(f"Missing full dataset directory: {data_root}")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"Missing clean CSV: {csv_path}")

df = pd.read_csv(csv_path)
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

print("rows:", len(df))
print("split counts:")
print(df["split"].astype(str).str.lower().value_counts().reindex(["train", "val", "test"], fill_value=0).to_string())
print("label distribution:")
print(pd.crosstab(df["label"], df["split"].astype(str).str.lower()).to_string())

random.seed(3407)
sample = df.sample(n=min(10, len(df)), random_state=3407)
for _, row in sample.iterrows():
    image_path = str(row["image_path"])
    path = image_path if os.path.isabs(image_path) else os.path.join(data_root, image_path)
    with Image.open(path) as image:
        image.verify()
    print("image OK:", row["sample_id"], path)

print("full data check: OK")
PY
}

main 2>&1 | tee logs/check_data.log
