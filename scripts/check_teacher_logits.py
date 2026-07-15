#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.data import ID_TO_LABEL, LABEL_TO_ID, read_crisismmd_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the complete MKAN train logits cache")
    parser.add_argument("--cache", default="teacher_cache/mkan_train_logits.pt")
    parser.add_argument("--csv_path", default="data/clean/task2_clean_consistent.csv")
    parser.add_argument("--report", default="outputs/cache_check/logits_cache_report.json")
    parser.add_argument("--expected_samples", type=int, default=6090)
    return parser.parse_args()


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate(args):
    if not os.path.isfile(args.cache):
        raise FileNotFoundError(f"Teacher logits cache not found: {args.cache}")
    raw = torch_load(args.cache)
    if not isinstance(raw, dict):
        raise ValueError("Cache must be a dictionary")
    required = {
        "sample_ids", "logits", "label_to_id", "id_to_label", "split",
        "teacher_checkpoint", "teacher_project_dir", "teacher_model_name", "export_config",
    }
    missing_fields = sorted(required - set(raw))
    if missing_fields:
        raise ValueError(f"Cache is missing fields: {missing_fields}")
    if str(raw["split"]).lower() != "train":
        raise ValueError(f"Cache split must be train, got {raw['split']!r}")

    sample_ids = [str(value) for value in raw["sample_ids"]]
    if len(sample_ids) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} sample_ids, got {len(sample_ids)}")
    if len(sample_ids) != len(set(sample_ids)):
        seen = set()
        duplicate = next(value for value in sample_ids if value in seen or seen.add(value))
        raise ValueError(f"Duplicate cache sample_id: {duplicate}")
    dataframe = read_crisismmd_csv(args.csv_path)
    csv_ids = dataframe.loc[
        dataframe["split"].astype(str).str.lower() == "train", "sample_id"
    ].astype(str).tolist()
    if len(csv_ids) != args.expected_samples:
        raise ValueError(f"CSV train split has {len(csv_ids)} samples, expected {args.expected_samples}")
    missing = sorted(set(csv_ids) - set(sample_ids))
    extra = sorted(set(sample_ids) - set(csv_ids))
    if missing:
        raise ValueError(f"Missing teacher logits for sample_id: {missing[0]}")
    if extra:
        raise ValueError(f"Extra teacher logits for sample_id: {extra[0]}")

    logits = torch.as_tensor(raw["logits"])
    if tuple(logits.shape) != (args.expected_samples, 5):
        raise ValueError(f"Logits shape must be [{args.expected_samples}, 5], got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Teacher logits contain NaN or Inf")
    label_to_id = {str(key): int(value) for key, value in raw["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in raw["id_to_label"].items()}
    if label_to_id != LABEL_TO_ID or id_to_label != ID_TO_LABEL:
        raise ValueError("Cache label mapping does not match the student's fixed five-class mapping")
    return {
        "status": "PASS",
        "cache": os.path.abspath(args.cache),
        "csv_path": os.path.abspath(args.csv_path),
        "split": "train",
        "sample_count": len(sample_ids),
        "unique_sample_count": len(set(sample_ids)),
        "missing_sample_count": 0,
        "extra_sample_count": 0,
        "logits_shape": list(logits.shape),
        "logits_finite": True,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "teacher_checkpoint": str(raw["teacher_checkpoint"]),
        "teacher_model_name": str(raw["teacher_model_name"]),
    }


def main() -> int:
    args = parse_args()
    try:
        report = validate(args)
        code = 0
    except Exception as exc:
        report = {
            "status": "FAIL",
            "cache": os.path.abspath(args.cache),
            "csv_path": os.path.abspath(args.csv_path),
            "error": str(exc),
        }
        code = 2
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"cache check report: {args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
