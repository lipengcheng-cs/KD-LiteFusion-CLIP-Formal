#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.data import LABEL_TO_ID, build_label_mapping, read_crisismmd_csv
from teacher_adapters.mkan_teacher_adapter import (
    checkpoint_top_level_fields,
    load_teacher,
    read_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a real MKAN-Refine teacher before cache export")
    parser.add_argument("--teacher_project_dir", default="")
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--teacher_config", default="")
    parser.add_argument("--csv_path", default="data/clean/task2_clean_consistent.csv")
    parser.add_argument("--image_root", default="data/CrisisMMD_v2.0")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def first_batch(loader):
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Teacher dataloader produced no batches") from exc


def main() -> int:
    args = parse_args()
    if not args.teacher_checkpoint or not os.path.isfile(args.teacher_checkpoint):
        print("Teacher checkpoint not found.", file=sys.stderr)
        print("Please provide a valid MKAN-Refine checkpoint path.", file=sys.stderr)
        return 2
    if not args.teacher_project_dir or not os.path.isdir(args.teacher_project_dir):
        print(f"Teacher project directory not found: {args.teacher_project_dir or 'NOT PROVIDED'}", file=sys.stderr)
        return 2
    if not args.teacher_config or not os.path.isfile(args.teacher_config):
        print(f"Teacher config not found: {args.teacher_config or 'NOT PROVIDED'}", file=sys.stderr)
        return 2

    try:
        checkpoint = read_checkpoint(args.teacher_checkpoint)
        fields = checkpoint_top_level_fields(checkpoint)
        print(f"checkpoint_top_level_fields: {fields}")
    except Exception as exc:
        print(f"Teacher checkpoint is not readable: {exc}", file=sys.stderr)
        return 2

    try:
        dataframe = read_crisismmd_csv(args.csv_path)
        label_to_id, _ = build_label_mapping(dataframe)
        if label_to_id != LABEL_TO_ID:
            raise ValueError(f"Student label mapping mismatch: {label_to_id}")
        train_df = dataframe[dataframe["split"].astype(str).str.lower() == "train"].head(2).copy()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter = load_teacher(
            args.teacher_project_dir,
            args.teacher_checkpoint,
            args.teacher_config,
            device,
        )
        loader = adapter.build_dataloader(train_df, args.image_root, batch_size=2, num_workers=args.num_workers)
        batch = first_batch(loader)
        adapter.model.eval()
        with torch.no_grad():
            logits = adapter.get_teacher_logits(batch)
        if adapter.model.training:
            raise RuntimeError("Teacher is not in eval mode")
        if logits.ndim != 2 or logits.shape[1] != 5:
            raise ValueError(f"Teacher output dimension must be 5, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Teacher logits contain NaN or Inf")
        print(f"teacher_model_name: {adapter.teacher_model_name}")
        print(f"teacher_label_to_id: {adapter.teacher_label_to_id}")
        print(f"student_output_label_to_id: {LABEL_TO_ID}")
        print(f"logits_shape: {tuple(logits.shape)}")
        print("teacher_eval_mode: OK")
        print("teacher_logits_finite: OK")
        print("MKAN teacher check: PASS")
        return 0
    except Exception as exc:
        print(f"MKAN teacher check: FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

