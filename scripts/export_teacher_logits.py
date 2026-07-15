#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.data import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    canonical_label,
    read_crisismmd_csv,
)
from kd_litefusion_mkan_teacher.utils import atomic_torch_save
from teacher_adapters.mkan_teacher_adapter import load_teacher


def parse_args():
    parser = argparse.ArgumentParser(description="Export sample-id aligned MKAN train logits")
    parser.add_argument("--teacher_project_dir", default="")
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--teacher_config", default="")
    parser.add_argument("--csv_path", default="data/clean/task2_clean_consistent.csv")
    parser.add_argument("--image_root", default="data/CrisisMMD_v2.0")
    parser.add_argument("--output", default="teacher_cache/mkan_train_logits.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--expected_samples", type=int, default=6090)
    return parser.parse_args()


def batch_sample_ids(batch) -> list[str]:
    if not isinstance(batch, dict) or "sample_id" not in batch:
        raise ValueError("Teacher runtime batches must contain sample_id")
    values = batch["sample_id"]
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return [str(value) for value in values]


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

    dataframe = read_crisismmd_csv(args.csv_path)
    train_df = dataframe[dataframe["split"].astype(str).str.lower() == "train"].copy()
    if len(train_df) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} train samples, found {len(train_df)}")
    train_df["label"] = train_df["label"].map(canonical_label)
    train_df["label_id"] = train_df["label"].map(LABEL_TO_ID)
    expected_ids = train_df["sample_id"].astype(str).tolist()
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Train CSV contains duplicate sample_id")
    expected_id_set = set(expected_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_teacher(
        args.teacher_project_dir,
        args.teacher_checkpoint,
        args.teacher_config,
        device,
    )
    loader = adapter.build_dataloader(
        train_df,
        args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logits_by_id = {}
    adapter.model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="export MKAN train logits"):
            sample_ids = batch_sample_ids(batch)
            logits = adapter.get_teacher_logits(batch).detach().cpu().float()
            if logits.shape != (len(sample_ids), 5):
                raise ValueError(
                    f"Teacher batch logits shape mismatch: ids={len(sample_ids)}, logits={tuple(logits.shape)}"
                )
            for sample_id, row_logits in zip(sample_ids, logits):
                if sample_id not in expected_id_set:
                    raise ValueError(f"Unexpected teacher sample_id: {sample_id}")
                if sample_id in logits_by_id:
                    raise ValueError(f"Duplicate teacher logits for sample_id: {sample_id}")
                logits_by_id[sample_id] = row_logits

    missing = [sample_id for sample_id in expected_ids if sample_id not in logits_by_id]
    extra = sorted(set(logits_by_id) - expected_id_set)
    if missing:
        raise ValueError(f"Missing teacher logits for sample_id: {missing[0]}")
    if extra:
        raise ValueError(f"Unexpected teacher sample_id: {extra[0]}")
    ordered_logits = torch.stack([logits_by_id[sample_id] for sample_id in expected_ids])
    if tuple(ordered_logits.shape) != (args.expected_samples, 5):
        raise ValueError(f"Final logits shape must be [{args.expected_samples}, 5]")
    if not torch.isfinite(ordered_logits).all():
        raise FloatingPointError("Teacher logits contain NaN or Inf")

    payload = {
        "sample_ids": expected_ids,
        "logits": ordered_logits,
        "label_to_id": dict(LABEL_TO_ID),
        "id_to_label": dict(ID_TO_LABEL),
        "split": "train",
        "teacher_checkpoint": os.path.abspath(args.teacher_checkpoint),
        "teacher_project_dir": os.path.abspath(args.teacher_project_dir),
        "teacher_model_name": adapter.teacher_model_name,
        "export_config": {
            "teacher_config": os.path.abspath(args.teacher_config),
            "csv_path": os.path.abspath(args.csv_path),
            "image_root": os.path.abspath(args.image_root),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "expected_samples": args.expected_samples,
        },
    }
    atomic_torch_save(payload, args.output)
    print(f"saved teacher logits cache: {args.output}")
    print(f"sample_ids: {len(expected_ids)}")
    print(f"logits_shape: {tuple(ordered_logits.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
