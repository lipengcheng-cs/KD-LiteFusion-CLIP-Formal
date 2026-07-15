#!/usr/bin/env python3
"""Export strict fixed-split formal teacher caches from frozen CLIP features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from formal_data import CachedFeatureDataset, STUDENT_LABEL_TO_ID, load_formal_csv
from model import MKANHead


SEEDS = [3407, 42, 2024]
NATIVE_TO_STUDENT_COLUMNS = [0, 1, 4, 3, 2]
IDENTITY = "mkan_refine_supplied_source_reproduction"
PROTOCOL = "student_fixed_split_6090_995_950"


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


def move_features(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("vision_tokens", "text_tokens", "vision_global", "text_global")
    }


def load_head(path: Path, device):
    payload = load_torch(path)
    if payload.get("teacher_identity") != IDENTITY:
        raise ValueError(f"Teacher identity mismatch: {path}")
    if payload.get("strict_b_spline_reproduction") is not False:
        raise ValueError(f"Invalid strict B-spline claim: {path}")
    if payload.get("training_protocol") != PROTOCOL:
        raise ValueError(f"Training protocol mismatch: {path}")
    model = MKANHead().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["paths"]["output_root"])
    selected_path = root / "reports" / "ensemble_selected_weights.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("test_used_for_selection") is not False:
        raise ValueError("Selected teacher strategy was not validation-only")
    weights = torch.tensor(
        [float(selected["selected_weights"][str(seed)]) for seed in SEEDS], dtype=torch.float32
    )
    if (weights < 0).any() or not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError(f"Invalid selected weights: {weights.tolist()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal teacher cache export")

    dataframe = load_formal_csv(
        config["data"]["csv_path"], config["data"]["expected_split_counts"]
    )
    train_df = dataframe[dataframe["split"] == "train"].copy()
    expected_ids = train_df["sample_id"].astype(str).tolist()
    train_payload = load_torch(root / "artifacts" / "clip_features" / "train_clip_features.pt")
    actual_ids = [str(value) for value in train_payload["sample_ids"]]
    if actual_ids != expected_ids:
        raise ValueError("Formal train feature cache IDs/order do not match the fixed student CSV")
    if train_payload.get("training_protocol") != PROTOCOL:
        raise ValueError("Formal train feature cache protocol mismatch")
    dataset = CachedFeatureDataset(train_payload)
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=int(config["training"]["num_workers"]), pin_memory=True,
    )
    device = torch.device("cuda")
    checkpoints = [root / "checkpoints" / f"ema_seed{seed}.pth" for seed in SEEDS]
    heads = [load_head(path, device) for path in checkpoints]
    device_weights = weights.to(device)

    sample_ids, logits_all, feature_all, gate_all = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="export formal fixed-split teacher cache"):
            features = move_features(batch, device)
            per_model = [head(**features) for head in heads]
            native_logits = sum(
                device_weights[index] * outputs["logits"] for index, outputs in enumerate(per_model)
            )
            fused_feature = sum(
                device_weights[index] * outputs["feature"] for index, outputs in enumerate(per_model)
            )
            gate = sum(
                device_weights[index] * outputs["gate"] for index, outputs in enumerate(per_model)
            )
            sample_ids.extend(str(value) for value in batch["sample_id"])
            logits_all.append(native_logits[:, NATIVE_TO_STUDENT_COLUMNS].float().cpu())
            feature_all.append(fused_feature.float().cpu())
            gate_all.append(gate.float().cpu())
    if sample_ids != expected_ids or len(set(sample_ids)) != 6090:
        raise ValueError("Exported cache sample IDs are not the exact 6,090 unique train IDs")
    logits = torch.cat(logits_all)
    feature = torch.cat(feature_all)
    gate = torch.cat(gate_all)
    if logits.shape != (6090, 5) or feature.shape != (6090, 768) or gate.shape != (6090, 768):
        raise ValueError(
            f"Invalid formal cache shapes: logits={logits.shape}, feature={feature.shape}, gate={gate.shape}"
        )
    if not all(torch.isfinite(value).all() for value in (logits, feature, gate)):
        raise ValueError("Formal teacher cache contains NaN/Inf")
    labels = torch.tensor(train_df["student_label_id"].astype(int).tolist(), dtype=torch.long)
    prototypes = torch.stack([feature[labels == class_id].mean(dim=0) for class_id in range(5)])
    if prototypes.shape != (5, 768) or not torch.isfinite(prototypes).all():
        raise ValueError("Invalid formal teacher prototypes")
    id_to_label = {index: label for label, index in STUDENT_LABEL_TO_ID.items()}
    common = {
        "sample_ids": sample_ids,
        "logits": logits,
        "label_to_id": dict(STUDENT_LABEL_TO_ID),
        "id_to_label": id_to_label,
        "split": "train",
        "teacher_checkpoints": [str(path.resolve()) for path in checkpoints],
        "selected_teacher_strategy": selected["selected_strategy"],
        "ensemble_weights": {str(seed): float(value) for seed, value in zip(SEEDS, weights.tolist())},
        "teacher_identity": IDENTITY,
        "teacher_display_name": "MKAN-Refine supplied-source reproduction teacher",
        "strict_b_spline_reproduction": False,
        "training_protocol": PROTOCOL,
        "data_audit_report": str(Path(config["paths"]["data_audit_report"]).resolve()),
        "source_csv": str(Path(config["data"]["csv_path"]).resolve()),
        "formal_teacher_config": str(config_path.resolve()),
    }
    cache_dir = root / "teacher_cache"
    atomic_save(common, cache_dir / "mkan_train_logits.pt")
    atomic_save(
        {**common, "feature": feature, "gate": gate, "prototypes": prototypes},
        cache_dir / "mkan_train_full.pt",
    )
    print(json.dumps({
        "logits": list(logits.shape), "feature": list(feature.shape),
        "gate": list(gate.shape), "prototypes": list(prototypes.shape),
        "selected_strategy": selected["selected_strategy"],
    }, indent=2))


if __name__ == "__main__":
    main()
