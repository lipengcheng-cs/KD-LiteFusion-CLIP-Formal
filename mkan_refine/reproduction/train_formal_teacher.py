#!/usr/bin/env python3
"""Train and evaluate the three fixed-split formal reproduction-teacher seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from formal_data import CachedFeatureDataset, NATIVE_LABEL_TO_ID
from model import MKANHead


IDENTITY = "mkan_refine_supplied_source_reproduction"
PROTOCOL = "student_fixed_split_6090_995_950"
NATIVE_LABELS = [label for label, _ in sorted(NATIVE_LABEL_TO_ID.items(), key=lambda item: item[1])]


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_features(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("vision_tokens", "text_tokens", "vision_global", "text_global")
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    sample_ids, labels, predictions, logits_all = [], [], [], []
    for batch in loader:
        outputs = model(**move_features(batch, device))
        logits = outputs["logits"].float().cpu()
        sample_ids.extend(str(value) for value in batch["sample_id"])
        labels.extend(batch["label"].tolist())
        predictions.extend(logits.argmax(dim=-1).tolist())
        logits_all.append(logits)
    return sample_ids, labels, predictions, torch.cat(logits_all)


def overall_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
    }


def evaluate(model, loader, device):
    sample_ids, labels, predictions, logits = predict(model, loader, device)
    return overall_metrics(labels, predictions), sample_ids, labels, predictions, logits


def checkpoint_payload(state_dict, seed, epoch, validation_metrics, config):
    return {
        "model_state_dict": state_dict,
        "label_to_id": dict(NATIVE_LABEL_TO_ID),
        "seed": int(seed),
        "epoch": int(epoch),
        "validation_metrics": validation_metrics,
        "training_config": config,
        "teacher_identity": IDENTITY,
        "strict_b_spline_reproduction": False,
        "training_protocol": PROTOCOL,
        "implementation_scope": (
            "MKAN-Refine supplied-source reproduction; supplied KANLinear is not a true B-spline KAN"
        ),
    }


def load_model_checkpoint(path: Path, device):
    payload = load_torch(path)
    if payload.get("teacher_identity") != IDENTITY or payload.get("training_protocol") != PROTOCOL:
        raise ValueError(f"Teacher identity/protocol mismatch in {path}")
    model = MKANHead().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval(), payload


def write_detailed_metrics(seed_dir: Path, labels, predictions) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(5)), zero_division=0
    )
    with open(seed_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for class_id, name in enumerate(NATIVE_LABELS):
            writer.writerow([class_id, name, precision[class_id], recall[class_id], f1[class_id], support[class_id]])
    matrix = confusion_matrix(labels, predictions, labels=list(range(5)))
    with open(seed_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *NATIVE_LABELS])
        for name, row in zip(NATIVE_LABELS, matrix.tolist()):
            writer.writerow([name, *row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["teacher_identity"] != IDENTITY or config["training_protocol"] != PROTOCOL:
        raise ValueError("Formal teacher identity or protocol is not the required value")
    if config.get("strict_b_spline_reproduction") is not False:
        raise ValueError("This supplied-source teacher must not claim strict B-spline reproduction")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal teacher training")
    device = torch.device("cuda")
    output_root = Path(config["paths"]["output_root"])
    feature_dir = output_root / "artifacts" / "clip_features"
    checkpoint_dir = output_root / "checkpoints"
    report_dir = output_root / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        split: load_torch(feature_dir / f"{split}_clip_features.pt")
        for split in ("train", "val", "test")
    }
    expected_counts = config["data"]["expected_split_counts"]
    for split, payload in payloads.items():
        if len(payload["sample_ids"]) != int(expected_counts[split]):
            raise ValueError(f"{split} cache count mismatch")
        if payload.get("split") != split or payload.get("training_protocol") != PROTOCOL:
            raise ValueError(f"{split} cache split/protocol mismatch")
        if len(set(str(value) for value in payload["sample_ids"])) != len(payload["sample_ids"]):
            raise ValueError(f"Duplicate sample IDs in {split} cache")
    if set(payloads["train"]["sample_ids"]) & set(payloads["val"]["sample_ids"]):
        raise ValueError("Train/validation feature IDs overlap")
    if set(payloads["train"]["sample_ids"]) & set(payloads["test"]["sample_ids"]):
        raise ValueError("Train/test feature IDs overlap")

    datasets = {split: CachedFeatureDataset(payload) for split, payload in payloads.items()}
    train_cfg = config["training"]
    class_weight = None
    if bool(train_cfg.get("class_weight", False)):
        counts = torch.bincount(payloads["train"]["labels"].long(), minlength=5).float()
        inverse = 1.0 / counts.clamp_min(1)
        class_weight = inverse.div(inverse.mean()).to(device)

    summary_rows = []
    for seed in [int(value) for value in train_cfg["seeds"]]:
        seed_everything(seed)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(seed)
        loaders = {
            "train": DataLoader(
                datasets["train"], batch_size=int(train_cfg["batch_size"]), shuffle=True,
                generator=generator, num_workers=int(train_cfg["num_workers"]), pin_memory=True,
            ),
            "val": DataLoader(
                datasets["val"], batch_size=int(train_cfg["batch_size"]), shuffle=False,
                num_workers=int(train_cfg["num_workers"]), pin_memory=True,
            ),
            "test": DataLoader(
                datasets["test"], batch_size=int(train_cfg["batch_size"]), shuffle=False,
                num_workers=int(train_cfg["num_workers"]), pin_memory=True,
            ),
        }
        model = MKANHead().to(device)
        ema = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(float(train_cfg["ema_decay"]))
        ).to(device)
        optimizer = AdamW(
            model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        best_weighted = (-1.0, -1.0)
        best_macro = (-1.0, -1.0)
        history = []
        epochs = int(train_cfg["epochs"])
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            progress = tqdm(loaders["train"], desc=f"formal teacher seed {seed} epoch {epoch}/{epochs}")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**move_features(batch, device))
                labels = batch["label"].to(device, non_blocking=True)
                loss = F.cross_entropy(outputs["logits"], labels, weight=class_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at seed={seed}, epoch={epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
                optimizer.step()
                ema.update_parameters(model)
                running_loss += float(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")
            val_metrics, *_ = evaluate(ema, loaders["val"], device)
            record = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(loaders["train"])),
                **val_metrics,
            }
            history.append(record)
            print(json.dumps({"seed": seed, **record}))
            state = copy.deepcopy(ema.module.state_dict())
            weighted_key = (val_metrics["weighted_f1"], val_metrics["macro_f1"])
            macro_key = (val_metrics["macro_f1"], val_metrics["weighted_f1"])
            if weighted_key > best_weighted:
                best_weighted = weighted_key
                atomic_torch_save(
                    checkpoint_payload(state, seed, epoch, val_metrics, config),
                    seed_dir / "best_weighted_f1.pt",
                )
            if macro_key > best_macro:
                best_macro = macro_key
                atomic_torch_save(
                    checkpoint_payload(state, seed, epoch, val_metrics, config),
                    seed_dir / "best_macro_f1.pt",
                )

        last_val_metrics, *_ = evaluate(ema, loaders["val"], device)
        atomic_torch_save(
            checkpoint_payload(copy.deepcopy(ema.module.state_dict()), seed, epochs, last_val_metrics, config),
            seed_dir / "last.pt",
        )
        best_model, best_payload = load_model_checkpoint(seed_dir / "best_weighted_f1.pt", device)
        val_metrics, val_sample_ids, val_labels, val_predictions, val_logits = evaluate(
            best_model, loaders["val"], device
        )
        test_metrics, sample_ids, labels, predictions, logits = evaluate(
            best_model, loaders["test"], device
        )
        val_record = {**val_metrics, "selected_epoch": best_payload["epoch"], "selection": "best_weighted_f1"}
        test_record = {**test_metrics, "selected_epoch": best_payload["epoch"], "selection": "best_weighted_f1"}
        atomic_json(history, seed_dir / "train_history.json")
        atomic_json(val_record, seed_dir / "val_metrics.json")
        atomic_json(test_record, seed_dir / "test_metrics.json")
        write_detailed_metrics(seed_dir, labels, predictions)
        with open(seed_dir / "val_predictions.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "true_native_id", "pred_native_id", *[f"logit_{x}" for x in NATIVE_LABELS]])
            for sample_id, label, prediction, row in zip(
                val_sample_ids, val_labels, val_predictions, val_logits.tolist()
            ):
                writer.writerow([sample_id, label, prediction, *row])
        with open(seed_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "true_native_id", "pred_native_id", *[f"logit_{x}" for x in NATIVE_LABELS]])
            for sample_id, label, prediction, row in zip(sample_ids, labels, predictions, logits.tolist()):
                writer.writerow([sample_id, label, prediction, *row])
        root_checkpoint = checkpoint_dir / f"ema_seed{seed}.pth"
        atomic_torch_save(best_payload, root_checkpoint)
        atomic_torch_save(best_payload, seed_dir / f"ema_seed{seed}.pth")
        summary_rows.append({"seed": seed, **val_record, **{f"test_{k}": v for k, v in test_metrics.items()}})

    with open(report_dir / "teacher_seed_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    atomic_json(
        {
            "teacher_identity": IDENTITY,
            "strict_b_spline_reproduction": False,
            "training_protocol": PROTOCOL,
            "seeds": summary_rows,
            "config_path": str(config_path.resolve()),
        },
        report_dir / "teacher_training_summary.json",
    )


if __name__ == "__main__":
    main()
