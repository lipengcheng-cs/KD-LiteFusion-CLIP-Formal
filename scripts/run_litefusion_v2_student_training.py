#!/usr/bin/env python3
"""Run one formal LiteFusion-v2 student experiment without knowledge distillation."""

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kd_litefusion_mkan_teacher.data import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    build_dataloaders,
    canonical_label,
    read_crisismmd_csv,
)
from kd_litefusion_mkan_teacher.litefusion_v2 import LiteFusionV2Config, LiteFusionV2Model, load_config
from kd_litefusion_mkan_teacher.litefusion_v2.profiling import parameter_breakdown, static_head_macs
from kd_litefusion_mkan_teacher.metrics import compute_metrics, confusion_matrix_df, per_class_metrics_df
from kd_litefusion_mkan_teacher.utils import atomic_torch_save, move_to_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--csv-path",
        default=str(PROJECT_ROOT / "data/clean/task2_clean_consistent.csv"),
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "data/CrisisMMD_v2.0"),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--class-weight-method",
        choices=("inverse_freq", "effective_num"),
        default="inverse_freq",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs != 10:
        raise ValueError("Formal student optimization runs require exactly 10 epochs")
    if args.batch_size != 8 or args.num_workers != 0:
        raise ValueError("Formal student optimization requires batch_size=8 and num_workers=0")
    if args.seed not in {3407, 42, 2024}:
        raise ValueError("Allowed formal seeds are 3407, 42, and 2024")
    if args.scheduler == "none" and args.warmup_epochs != 0:
        raise ValueError("scheduler=none requires warmup_epochs=0")
    if args.scheduler == "cosine" and args.warmup_epochs != 1:
        raise ValueError("Formal cosine runs require warmup_epochs=1")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if args.lr <= 0 or args.min_lr < 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("Optimizer and gradient-clipping values must be valid and non-negative")
    if args.scheduler == "cosine" and args.min_lr >= args.lr:
        raise ValueError("min_lr must be lower than lr for cosine scheduling")


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def atomic_json_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def atomic_text_save(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def ensure_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def gpu_snapshot() -> Dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    query = "name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu"
    output = subprocess.check_output(
        ("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"),
        text=True,
    ).strip()
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    return {
        "available": True,
        "name": values[0],
        "driver_version": values[1],
        "memory_total_mb": float(values[2]),
        "memory_used_mb": float(values[3]),
        "utilization_percent": float(values[4]),
        "temperature_c": float(values[5]),
    }


def environment_snapshot(commit: str) -> Dict[str, object]:
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_before": gpu_snapshot(),
        "command": shlex.join(sys.argv),
    }


def build_model_config(args: argparse.Namespace) -> LiteFusionV2Config:
    config = load_config(args.config)
    values = config.to_dict()
    values["dropout"] = float(args.dropout)
    return LiteFusionV2Config.from_dict(values)


def class_weights(csv_path: str, method: str, device: torch.device) -> torch.Tensor:
    frame = read_crisismmd_csv(csv_path)
    train_labels = frame.loc[
        frame["split"].astype(str).str.lower() == "train", "label"
    ].map(canonical_label)
    counts = torch.ones(len(LABEL_TO_ID), dtype=torch.float32)
    for label, count in train_labels.value_counts().items():
        counts[LABEL_TO_ID[label]] = float(count)
    if method == "effective_num":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    else:
        weights = 1.0 / counts
    return (weights / weights.mean().clamp_min(1e-8)).to(device)


def validate_output_contract(outputs: Mapping[str, torch.Tensor], batch_size: int, feature_dim: int) -> None:
    required = {"logits", "feature", "gate", "image_feature", "text_feature"}
    missing = sorted(required - set(outputs))
    if missing:
        raise ValueError(f"LiteFusion-v2 output is missing fields: {missing}")
    for key in ("feature", "gate", "image_feature", "text_feature"):
        if tuple(outputs[key].shape) != (batch_size, feature_dim):
            raise ValueError(f"{key} shape mismatch: {tuple(outputs[key].shape)}")
        if not torch.isfinite(outputs[key]).all():
            raise FloatingPointError(f"{key} contains NaN or Inf")
    if tuple(outputs["logits"].shape) != (batch_size, len(LABEL_TO_ID)):
        raise ValueError(f"logits shape mismatch: {tuple(outputs['logits'].shape)}")
    if not torch.isfinite(outputs["logits"]).all():
        raise FloatingPointError("logits contain NaN or Inf")


@torch.inference_mode()
def evaluate_validation(model, loader, device) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    sample_ids: List[str] = []
    labels: List[int] = []
    predictions: List[int] = []
    for batch in tqdm(loader, desc="validation", leave=False):
        sample_ids.extend(batch["sample_id"])
        batch = move_to_device(batch, device)
        forbidden = sorted(key for key in batch if key.startswith("teacher_"))
        if forbidden:
            raise ValueError(f"Validation batch unexpectedly contains teacher fields: {forbidden}")
        outputs = model(batch["images"], batch["text_tokens"], return_dict=True)
        validate_output_contract(outputs, batch["labels"].shape[0], model.config.feature_dim)
        predicted = outputs["logits"].argmax(dim=-1)
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
    metrics = compute_metrics(labels, predictions)
    predictions_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "prediction": predictions,
            "label_name": [ID_TO_LABEL[int(value)] for value in labels],
            "prediction_name": [ID_TO_LABEL[int(value)] for value in predictions],
        }
    )
    return metrics, predictions_frame


def checkpoint_payload(
    args: argparse.Namespace,
    model: LiteFusionV2Model,
    epoch: int,
    metrics: Mapping[str, float],
    commit: str,
) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "validation_metrics": dict(metrics),
        "student_state_dict": model.student_state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": {
            "config_name": args.config_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "class_weight_method": args.class_weight_method,
            "label_smoothing": args.label_smoothing,
            "max_grad_norm": args.max_grad_norm,
            "kd": False,
            "teacher_cache": None,
            "selection_split": "val",
        },
        "label_to_id": dict(LABEL_TO_ID),
        "id_to_label": dict(ID_TO_LABEL),
        "git_commit": commit,
    }


def save_validation_artifacts(output_dir: Path, metrics: Mapping[str, float], predictions: pd.DataFrame) -> None:
    atomic_json_save(dict(metrics), output_dir / "val_metrics.json")
    predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    labels = predictions["label"].astype(int).tolist()
    predicted = predictions["prediction"].astype(int).tolist()
    per_class_metrics_df(labels, predicted, ID_TO_LABEL).to_csv(
        output_dir / "val_per_class_metrics.csv", index=False
    )
    confusion_matrix_df(labels, predicted, ID_TO_LABEL).to_csv(
        output_dir / "val_confusion_matrix.csv"
    )


def make_scheduler(args: argparse.Namespace, optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    min_ratio = args.min_lr / args.lr

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1, step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=factor)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    ensure_fresh_output(output_dir)
    set_seed(args.seed)
    commit = git_commit()
    command = shlex.join(sys.argv)
    atomic_text_save(command + "\n", output_dir / "command.txt")
    atomic_text_save(commit + "\n", output_dir / "git_commit.txt")
    atomic_json_save(environment_snapshot(commit), output_dir / "environment.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal student training requires CUDA")
    model_config = build_model_config(args)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model": model_config.to_dict(),
                "training": {
                    "config_name": args.config_name,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "optimizer": "AdamW",
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "scheduler": args.scheduler,
                    "warmup_epochs": args.warmup_epochs,
                    "min_lr": args.min_lr,
                    "class_weight_method": args.class_weight_method,
                    "label_smoothing": args.label_smoothing,
                    "dropout": args.dropout,
                    "max_grad_norm": args.max_grad_norm,
                    "checkpoint_primary": "validation_weighted_f1",
                    "checkpoint_secondary": "validation_macro_f1",
                },
                "data": {
                    "csv_path": str(Path(args.csv_path).resolve()),
                    "image_root": str(Path(args.image_root).resolve()),
                    "teacher_cache": None,
                    "selection_split": "val",
                    "test_evaluation": False,
                },
                "safety": {"kd": False, "clip_frozen": True, "overwrite": False},
            },
            handle,
            sort_keys=False,
        )

    model = LiteFusionV2Model(model_config, device=device, load_clip=True).to(device)
    if not model.freeze_clip or any(parameter.requires_grad for parameter in model.clip.parameters()):
        raise AssertionError("OpenAI CLIP must be frozen")
    loaders, label_to_id, id_to_label, teacher_cache = build_dataloaders(
        csv_path=args.csv_path,
        image_root=args.image_root,
        preprocess=model.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        teacher_cache_path=None,
    )
    if teacher_cache is not None:
        raise AssertionError("w/o KD training must not construct a teacher cache")
    if label_to_id != LABEL_TO_ID or id_to_label != ID_TO_LABEL:
        raise ValueError("Dataset does not match the fixed five-class mapping")
    if "train" not in loaders or "val" not in loaders:
        raise ValueError("CSV must contain train and val; fallback to test is forbidden")
    train_loader = loaders["train"]
    validation_loader = loaders["val"]

    weights = class_weights(args.csv_path, args.class_weight_method, device)
    optimizer = AdamW(list(model.head_parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args, optimizer, len(train_loader))
    params = parameter_breakdown(model)
    macs = static_head_macs(model, batch_size=1, device=device)
    atomic_json_save(
        {
            "head_params": params,
            "head_macs": macs,
            "class_weights": [float(value) for value in weights.detach().cpu().tolist()],
        },
        output_dir / "static_profile.json",
    )

    per_epoch_dir = output_dir / "per_epoch_validation"
    per_epoch_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_weighted_key = (-1.0, -1.0)
    best_macro_key = (-1.0, -1.0)
    best_weighted_epoch = None
    best_macro_epoch = None
    best_weighted_metrics = None
    best_weighted_predictions = None
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if model.clip.training:
            raise AssertionError("Frozen CLIP must remain in eval mode")
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"{model_config.name} {args.config_name} epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = move_to_device(batch, device)
            forbidden = sorted(key for key in batch if key.startswith("teacher_"))
            if forbidden:
                raise ValueError(f"w/o KD training batch contains teacher fields: {forbidden}")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["images"], batch["text_tokens"], return_dict=True)
            validate_output_contract(outputs, batch["labels"].shape[0], model_config.feature_dim)
            loss = F.cross_entropy(
                outputs["logits"],
                batch["labels"],
                weight=weights,
                label_smoothing=args.label_smoothing,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(model.head_parameters()), args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Gradient norm is NaN or Inf")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        validation_metrics, validation_predictions = evaluate_validation(model, validation_loader, device)
        labels = validation_predictions["label"].astype(int).tolist()
        predicted = validation_predictions["prediction"].astype(int).tolist()
        per_class = per_class_metrics_df(labels, predicted, ID_TO_LABEL)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            **{f"val_{key}": float(value) for key, value in validation_metrics.items()},
        }
        history.append(epoch_row)
        atomic_json_save(history, output_dir / "train_history.json")
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
        atomic_json_save(
            {**epoch_row, "class_weight_method": args.class_weight_method},
            per_epoch_dir / f"epoch_{epoch:02d}_metrics.json",
        )
        per_class.to_csv(per_epoch_dir / f"epoch_{epoch:02d}_per_class_metrics.csv", index=False)
        payload = checkpoint_payload(args, model, epoch, validation_metrics, commit)
        weighted_key = (validation_metrics["weighted_f1"], validation_metrics["macro_f1"])
        macro_key = (validation_metrics["macro_f1"], validation_metrics["weighted_f1"])
        if weighted_key > best_weighted_key:
            best_weighted_key = weighted_key
            best_weighted_epoch = epoch
            best_weighted_metrics = dict(validation_metrics)
            best_weighted_predictions = validation_predictions.copy()
            atomic_torch_save(payload, str(output_dir / "best_weighted_f1.pt"))
        if macro_key > best_macro_key:
            best_macro_key = macro_key
            best_macro_epoch = epoch
            atomic_torch_save(payload, str(output_dir / "best_macro_f1.pt"))
        atomic_torch_save(payload, str(output_dir / "last.pt"))
        print(json.dumps(epoch_row, sort_keys=True))

    if best_weighted_metrics is None or best_weighted_predictions is None:
        raise RuntimeError("No validation result was produced")
    save_validation_artifacts(output_dir, best_weighted_metrics, best_weighted_predictions)
    peak_memory_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    summary = {
        "candidate": model_config.name,
        "config_name": args.config_name,
        "seed": args.seed,
        "best_weighted_epoch": int(best_weighted_epoch),
        "best_macro_epoch": int(best_macro_epoch),
        "val_accuracy": float(best_weighted_metrics["accuracy"]),
        "val_weighted_f1": float(best_weighted_metrics["weighted_f1"]),
        "val_macro_f1": float(best_weighted_metrics["macro_f1"]),
        "val_precision": float(best_weighted_metrics["precision"]),
        "val_recall": float(best_weighted_metrics["recall"]),
        "head_params": int(params["full_head"]),
        "head_macs": int(macs["full_head"]),
        "peak_gpu_memory_mb": peak_memory_mb,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "class_weight_method": args.class_weight_method,
        "dropout": args.dropout,
        "kd": False,
        "teacher_cache": None,
        "selection_split": "val",
        "test_evaluation": False,
        "git_commit": commit,
        "gpu_after": gpu_snapshot(),
    }
    atomic_json_save(summary, output_dir / "training_summary.json")
    atomic_text_save("complete\n", output_dir / "COMPLETED")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
