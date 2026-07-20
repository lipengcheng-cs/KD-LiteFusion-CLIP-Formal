#!/usr/bin/env python3
"""Validation-only training for the strict MKAN paper-protocol baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mkan_refine.paper_reproduction_v2.metrics import classification_metrics, selection_key
from mkan_refine.paper_reproduction_v2.model import MKANPaperHeadV2, SplineConfig


class MemmapFeatureDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        metadata = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "PASS":
            raise ValueError(f"feature cache is not PASS: {split_dir}")
        self.metadata = metadata
        self.vision_tokens = np.load(split_dir / "vision_tokens.npy", mmap_mode="r")
        self.text_tokens = np.load(split_dir / "text_tokens.npy", mmap_mode="r")
        self.vision_global = np.load(split_dir / "vision_global.npy", mmap_mode="r")
        self.text_global = np.load(split_dir / "text_global.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "labels.npy", mmap_mode="r")
        lengths = {len(value) for value in (self.vision_tokens, self.text_tokens, self.vision_global, self.text_global, self.labels)}
        if lengths != {int(metadata["count"])}:
            raise ValueError(f"cache length mismatch: {split_dir}, lengths={lengths}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Copies avoid writable-memmap warnings and isolate DataLoader batches.
        return {
            "vision_tokens": torch.from_numpy(np.array(self.vision_tokens[index], copy=True)),
            "text_tokens": torch.from_numpy(np.array(self.text_tokens[index], copy=True)),
            "vision_global": torch.from_numpy(np.array(self.vision_global[index], copy=True)),
            "text_global": torch.from_numpy(np.array(self.text_global[index], copy=True)),
            "label": int(self.labels[index]),
        }


class ModelEMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        target = self.module.state_dict()
        for key, value in target.items():
            current = source[key].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                value.copy_(current)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state, strict=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=False)
        for key, value in batch.items()
        if key != "label"
    }, batch["label"].to(device)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions, logits = [], [], []
    for batch in loader:
        features, target = move_batch(batch, device)
        output = model(**features)["logits"]
        labels.extend(target.cpu().tolist())
        predictions.extend(output.argmax(dim=1).cpu().tolist())
        logits.append(output.float().cpu())
    metrics = classification_metrics(labels, predictions)
    return metrics, torch.cat(logits, dim=0), torch.tensor(labels, dtype=torch.long)


def atomic_torch_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model_state, epoch, metrics, config, seed, cache_metadata, architecture):
    return {
        "model_state_dict": model_state,
        "epoch": int(epoch),
        "validation_metrics": metrics,
        "config": config,
        "seed": int(seed),
        "cache_metadata": cache_metadata,
        "architecture": architecture,
        "teacher_identity": "mkan_refine_paper_reproduction_v2_true_b_spline",
        "strict_b_spline_reproduction": True,
        "test_used_for_selection": False,
    }


def plot_history(history: list[dict], output: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], color="tab:red")
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(output / "training_loss.png", dpi=200); plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["raw_val_weighted_f1"] for row in history], label="raw")
    plt.plot(epochs, [row["ema_val_weighted_f1"] for row in history], label="EMA")
    plt.xlabel("Epoch"); plt.ylabel("Validation Weighted-F1"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(output / "validation_weighted_f1.png", dpi=200); plt.close()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"]).resolve()
    output_dir = (root / config["output_dir"] / f"seed_{args.seed}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "COMPLETE.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") == "PASS" and int(complete.get("epochs", -1)) == int(config["training"]["epochs"]):
            print(f"SKIP_COMPLETE {output_dir}")
            return
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 5 * 2**30:
        raise RuntimeError(f"root disk free space below 5 GiB: {free_bytes / 2**30:.2f} GiB")

    cache_root = Path(config["feature_cache_root"]).resolve()
    datasets = {split: MemmapFeatureDataset(cache_root / split) for split in ("train", "val")}
    if len(datasets["train"]) != 5119 or len(datasets["val"]) != 1097:
        raise ValueError("paper split count mismatch")
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=False, generator=generator, drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=False, drop_last=False,
        ),
    }
    set_seed(args.seed)
    device = torch.device("cuda:0")
    spline = SplineConfig(**config["model"]["spline"])
    model = MKANPaperHeadV2(
        dim=768,
        num_classes=5,
        classifier_hidden=int(config["model"]["classifier_hidden"]),
        dropout=float(config["model"]["dropout"]),
        spline=spline,
        share_attention_scorer=bool(config["model"]["share_attention_scorer"]),
    ).to(device).float()
    ema = ModelEMA(model, float(config["training"]["ema_decay"]))
    optimizer = AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    def lr_lambda(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    counts = np.bincount(np.asarray(datasets["train"].labels), minlength=5).astype(np.float64)
    inverse = 1.0 / counts
    class_weights = torch.tensor(inverse / inverse.mean(), device=device, dtype=torch.float32)
    history = []
    best_raw = (-1.0, -1.0, -1.0)
    best_ema = (-1.0, -1.0, -1.0)
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if last_path.is_file() and bool(config["training"].get("resume", True)):
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["raw_model_state_dict"], strict=True)
        ema.load_state_dict(payload["ema_model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = payload["history"]
        best_raw = tuple(payload["best_raw_key"])
        best_ema = tuple(payload["best_ema_key"])
        start_epoch = int(payload["epoch"]) + 1

    architecture = {
        "name": "MKANPaperHeadV2",
        "strict_b_spline": True,
        "spline": config["model"]["spline"],
        "share_attention_scorer": config["model"]["share_attention_scorer"],
        "classifier_hidden": config["model"]["classifier_hidden"],
        "head_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    cache_metadata = {split: datasets[split].metadata for split in datasets}
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        train_labels, train_predictions = [], []
        progress = tqdm(loaders["train"], desc=f"paper v2 seed {args.seed} epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            features, target = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(**features)
            loss = F.cross_entropy(
                model_output["logits"], target, weight=class_weights,
                label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
            )
            regularization = float(config["training"].get("spline_regularization", 0.0))
            if regularization:
                loss = loss + regularization * model.regularization_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            gradient_check_interval = int(config["training"].get("gradient_check_interval", 50))
            if step == 1 or step % gradient_check_interval == 0:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite gradient: {name}")
            clip_norm = float(config["training"].get("gradient_clip_norm", 0.0))
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            ema.update(model)
            total_loss += float(loss.item()) * target.size(0)
            train_labels.extend(target.detach().cpu().tolist())
            train_predictions.extend(model_output["logits"].argmax(1).detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_metrics = classification_metrics(train_labels, train_predictions)
        raw_metrics, raw_logits, val_labels = evaluate(model, loaders["val"], device)
        ema_metrics, ema_logits, ema_labels = evaluate(ema.module, loaders["val"], device)
        if not torch.equal(val_labels, ema_labels):
            raise RuntimeError("raw and EMA validation label order differs")
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(datasets["train"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_accuracy": train_metrics["accuracy"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "raw_val_accuracy": raw_metrics["accuracy"],
            "raw_val_weighted_f1": raw_metrics["weighted_f1"],
            "raw_val_macro_f1": raw_metrics["macro_f1"],
            "ema_val_accuracy": ema_metrics["accuracy"],
            "ema_val_weighted_f1": ema_metrics["weighted_f1"],
            "ema_val_macro_f1": ema_metrics["macro_f1"],
        }
        history.append(row)
        (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        raw_key, ema_key = selection_key(raw_metrics), selection_key(ema_metrics)
        if raw_key > best_raw:
            best_raw = raw_key
            atomic_torch_save(
                checkpoint_payload(model.state_dict(), epoch, raw_metrics, config, args.seed, cache_metadata, architecture),
                output_dir / "best_raw.pt",
            )
            torch.save({"logits": raw_logits, "labels": val_labels}, output_dir / "best_raw_validation_logits.pt")
        if ema_key > best_ema:
            best_ema = ema_key
            atomic_torch_save(
                checkpoint_payload(ema.state_dict(), epoch, ema_metrics, config, args.seed, cache_metadata, architecture),
                output_dir / "best_ema.pt",
            )
            torch.save({"logits": ema_logits, "labels": val_labels}, output_dir / "best_ema_validation_logits.pt")
        last_payload = {
            "epoch": epoch,
            "raw_model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_raw_key": best_raw,
            "best_ema_key": best_ema,
            "config": config,
            "seed": args.seed,
            "test_evaluated": False,
        }
        atomic_torch_save(last_payload, last_path)
        print(json.dumps({"epoch": epoch, "raw": raw_metrics, "ema": ema_metrics}))

    with (output_dir / "raw_vs_ema.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    plot_history(history, output_dir)
    complete = {
        "status": "PASS",
        "seed": args.seed,
        "epochs": epochs,
        "best_raw_validation_key": best_raw,
        "best_ema_validation_key": best_ema,
        "test_evaluated": False,
        "selection_split": "validation",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete))


if __name__ == "__main__":
    main()
