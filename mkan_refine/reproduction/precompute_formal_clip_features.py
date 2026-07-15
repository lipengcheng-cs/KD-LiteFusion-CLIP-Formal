#!/usr/bin/env python3
"""Precompute frozen OpenAI CLIP features for the fixed student splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import clip
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from formal_data import (
    NATIVE_LABEL_TO_ID,
    RawFormalDataset,
    load_formal_csv,
    sample_id_hash,
    sha256_file,
)
from model import OpenAIClipTokenEncoder


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cache_matches(path: Path, expected_ids, csv_sha: str, split: str) -> bool:
    if not path.is_file():
        return False
    payload = load_torch(path)
    actual_ids = [str(value) for value in payload.get("sample_ids", [])]
    return (
        actual_ids == list(expected_ids)
        and payload.get("split") == split
        and payload.get("source_csv_sha256") == csv_sha
        and payload.get("training_protocol") == "student_fixed_split_6090_995_950"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    output_root = Path(config["paths"]["output_root"])
    feature_dir = output_root / "artifacts" / "clip_features"
    report_dir = output_root / "reports"
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    audit_path = Path(config["paths"]["data_audit_report"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("formal_teacher_required"):
        raise RuntimeError("Audit does not require a formal fixed-split teacher; refusing unexpected run")

    csv_path = data_cfg["csv_path"]
    dataframe = load_formal_csv(csv_path, data_cfg["expected_split_counts"])
    csv_sha = sha256_file(csv_path)
    split_ids = {
        split: dataframe.loc[dataframe["split"] == split, "sample_id"].astype(str).tolist()
        for split in ("train", "val", "test")
    }
    reuse_report = {
        "legacy_feature_dir": config["paths"].get("legacy_feature_dir"),
        "reused": False,
        "reason": (
            "Legacy caches were built from the 5119/1097/1098 teacher split. "
            "The formal protocol requires exact 6090/995/950 student split IDs, so no legacy cache is reused."
        ),
        "formal_split_id_hashes": {split: sample_id_hash(ids) for split, ids in split_ids.items()},
    }
    (report_dir / "feature_cache_reuse_audit.json").write_text(
        json.dumps(reuse_report, indent=2) + "\n", encoding="utf-8"
    )

    destinations = {split: feature_dir / f"{split}_clip_features.pt" for split in split_ids}
    if all(cache_matches(destinations[s], split_ids[s], csv_sha, s) for s in split_ids):
        print("All formal CLIP feature caches already match exact IDs, splits, and CSV hash; skipping.")
        return

    clip_path = Path(config["clip"]["checkpoint"])
    if not clip_path.is_file():
        raise FileNotFoundError(clip_path)
    if config["clip"].get("allow_network_download", False):
        raise ValueError("Formal configuration must prohibit network downloads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal CLIP feature precomputation")
    device = torch.device("cuda")
    clip_model, _ = clip.load(str(clip_path), device=device, jit=False)
    clip_model.requires_grad_(False).eval()
    encoder = OpenAIClipTokenEncoder(clip_model).to(device).eval()
    pre_cfg = config["precompute"]

    for split in ("train", "val", "test"):
        split_df = dataframe[dataframe["split"] == split].copy()
        expected_ids = split_ids[split]
        dataset = RawFormalDataset(split_df, data_cfg["image_root"], clip.tokenize)
        loader = DataLoader(
            dataset,
            batch_size=int(pre_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(pre_cfg["num_workers"]),
            pin_memory=True,
        )
        sample_ids, labels = [], []
        vt_all, tt_all, vg_all, tg_all = [], [], [], []
        for batch in tqdm(loader, desc=f"formal CLIP features: {split}"):
            outputs = encoder.encode(
                batch["pixel_values"].to(device, non_blocking=True),
                batch["input_ids"].to(device, non_blocking=True),
            )
            vt, tt, vg, tg = (value.detach().cpu().half() for value in outputs)
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels.append(batch["label"].cpu())
            vt_all.append(vt)
            tt_all.append(tt)
            vg_all.append(vg)
            tg_all.append(tg)
        if sample_ids != expected_ids:
            raise ValueError(f"{split} feature IDs do not exactly match the fixed CSV order")
        count = len(sample_ids)
        payload = {
            "sample_ids": sample_ids,
            "labels": torch.cat(labels),
            "vision_tokens": torch.cat(vt_all),
            "text_tokens": torch.cat(tt_all),
            "vision_global": torch.cat(vg_all),
            "text_global": torch.cat(tg_all),
            "split": split,
            "sample_id_hash": sample_id_hash(sample_ids),
            "label_to_id": dict(NATIVE_LABEL_TO_ID),
            "source_csv": str(Path(csv_path).resolve()),
            "source_csv_sha256": csv_sha,
            "image_root": str(Path(data_cfg["image_root"]).resolve()),
            "clip_checkpoint": str(clip_path.resolve()),
            "teacher_identity": config["teacher_identity"],
            "strict_b_spline_reproduction": False,
            "training_protocol": config["training_protocol"],
        }
        if payload["vision_tokens"].shape != (count, 577, 768):
            raise ValueError(f"Unexpected vision token shape: {payload['vision_tokens'].shape}")
        if payload["text_tokens"].shape != (count, 77, 768):
            raise ValueError(f"Unexpected text token shape: {payload['text_tokens'].shape}")
        if not all(
            torch.isfinite(payload[key]).all()
            for key in ("vision_tokens", "text_tokens", "vision_global", "text_global")
        ):
            raise ValueError(f"Non-finite values in {split} features")
        atomic_save(payload, destinations[split])
        print(f"saved {destinations[split]}: {count} samples")


if __name__ == "__main__":
    main()
