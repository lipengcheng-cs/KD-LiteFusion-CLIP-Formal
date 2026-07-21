#!/usr/bin/env python3
"""Validation-only LiteFusion-v2 screening (seed 3407, four epochs, no KD)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
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
from kd_litefusion_mkan_teacher.litefusion_v2 import CANDIDATE_NAMES, LiteFusionV2Model, load_config
from kd_litefusion_mkan_teacher.litefusion_v2.profiling import (
    parameter_breakdown,
    static_head_macs,
)
from kd_litefusion_mkan_teacher.metrics import compute_metrics, confusion_matrix_df, per_class_metrics_df
from kd_litefusion_mkan_teacher.utils import atomic_torch_save, move_to_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/litefusion_v2")
    parser.add_argument(
        "--csv-path",
        default=str(PROJECT_ROOT / "data/clean/task2_clean_consistent.csv"),
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "data/CrisisMMD_v2.0"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/litefusion_v2/screening_seed3407_4ep"),
    )
    parser.add_argument(
        "--benchmark-results",
        default=str(PROJECT_ROOT / "outputs/litefusion_v2/benchmark_fp32/benchmark_results.json"),
        help="Completed formal FP32 benchmark JSON. Screening reuses its batch-1 head metrics.",
    )
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATE_NAMES))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    return parser.parse_args()


def validate_protocol(args: argparse.Namespace) -> None:
    if args.seed != 3407:
        raise ValueError("LiteFusion-v2 first-round screening is fixed to seed=3407")
    if args.epochs != 4:
        raise ValueError("LiteFusion-v2 first-round screening is fixed to four epochs")
    unknown = sorted(set(args.candidates) - set(CANDIDATE_NAMES))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    if args.batch_size != 8 or args.num_workers != 0:
        raise ValueError("Formal screening requires batch_size=8 and num_workers=0")


def load_benchmark_summary(path: str, candidates: Sequence[str]) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark JSON must include metadata and results: {path}")
    fairness = payload.get("metadata", {}).get("fairness", {})
    if fairness.get("passed") is not True:
        raise ValueError("Benchmark fairness check did not pass; screening is blocked")
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Invalid benchmark results payload: {path}")
    summary: Dict[str, Dict] = {}
    for row in rows:
        if (
            row.get("candidate") in candidates
            and row.get("mode") == "head_only"
            and int(row.get("batch_size", -1)) == 1
            and row.get("name") in {"full_model", "full_head"}
        ):
            summary[row["candidate"]] = dict(row)
    missing = sorted(set(candidates) - set(summary))
    if missing:
        raise ValueError(f"Formal batch-1 head benchmark is missing candidates: {missing}")
    return summary


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def atomic_json_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def class_weights(csv_path: str, device: torch.device) -> torch.Tensor:
    frame = read_crisismmd_csv(csv_path)
    train_labels = frame.loc[
        frame["split"].astype(str).str.lower() == "train", "label"
    ].map(canonical_label)
    counts = torch.ones(len(LABEL_TO_ID), dtype=torch.float32)
    for label, count in train_labels.value_counts().items():
        counts[LABEL_TO_ID[label]] = float(count)
    weights = 1.0 / counts
    return (weights / weights.mean().clamp_min(1e-8)).to(device)


def validate_output_contract(outputs: Mapping[str, torch.Tensor], batch_size: int, feature_dim: int) -> None:
    required = {"logits", "feature", "gate", "image_feature", "text_feature"}
    missing = sorted(required - set(outputs))
    if missing:
        raise ValueError(f"LiteFusion-v2 output is missing fields: {missing}")
    expected_feature = (batch_size, feature_dim)
    for key in ("feature", "gate", "image_feature", "text_feature"):
        if tuple(outputs[key].shape) != expected_feature:
            raise ValueError(f"{key} shape mismatch: {tuple(outputs[key].shape)} != {expected_feature}")
    if tuple(outputs["logits"].shape) != (batch_size, len(LABEL_TO_ID)):
        raise ValueError(f"logits shape mismatch: {tuple(outputs['logits'].shape)}")


@torch.inference_mode()
def evaluate_validation(model, loader, device) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    sample_ids: List[str] = []
    labels: List[int] = []
    predictions: List[int] = []
    for batch in tqdm(loader, desc="validation", leave=False):
        sample_ids.extend(batch["sample_id"])
        batch = move_to_device(batch, device)
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


def checkpoint_payload(model, epoch: int, metrics: Mapping[str, float], commit: str) -> Dict:
    return {
        "epoch": int(epoch),
        "validation_metrics": dict(metrics),
        "student_state_dict": model.student_state_dict(),
        "config": model.config.to_dict(),
        "label_to_id": dict(LABEL_TO_ID),
        "id_to_label": dict(ID_TO_LABEL),
        "git_commit": commit,
        "protocol": {"seed": 3407, "epochs": 4, "kd": False, "selection_split": "val"},
    }


def save_validation_artifacts(
    output_dir: Path,
    metrics: Mapping[str, float],
    predictions: pd.DataFrame,
) -> None:
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


def train_candidate(
    args: argparse.Namespace,
    candidate: str,
    device: torch.device,
    benchmark: Mapping[str, object],
) -> Dict:
    config_path = Path(args.config_dir) / f"{candidate}.yaml"
    config = load_config(str(config_path))
    if config.name != candidate or config.interaction_rank != 32:
        raise ValueError(f"Invalid first-round candidate config: {config.to_dict()}")
    output_dir = Path(args.output_dir) / candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model": config.to_dict(),
                "protocol": {"seed": 3407, "epochs": 4, "kd": False, "selection_split": "val"},
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")

    model = LiteFusionV2Model(config, device=device, load_clip=True).to(device)
    loaders, label_to_id, id_to_label, teacher_cache = build_dataloaders(
        csv_path=args.csv_path,
        image_root=args.image_root,
        preprocess=model.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        teacher_cache_path=None,
    )
    if teacher_cache is not None:
        raise AssertionError("w/o KD screening must not construct a teacher cache")
    if label_to_id != LABEL_TO_ID or id_to_label != ID_TO_LABEL:
        raise ValueError("Dataset does not match the fixed five-class mapping")
    if "train" not in loaders:
        raise ValueError("CSV must contain split == train")
    if "val" not in loaders:
        raise ValueError("CSV must contain split == val; fallback to another split is forbidden")
    train_loader = loaders["train"]
    validation_loader = loaders["val"]

    optimizer = AdamW(model.head_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(args.csv_path, device)
    history: List[Dict] = []
    best_weighted = (-1.0, -1.0)
    best_macro = (-1.0, -1.0)
    best_epoch = None
    best_validation = None
    best_predictions = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        if model.clip.training:
            raise AssertionError("Frozen CLIP must remain in eval mode")
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"{candidate} epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = move_to_device(batch, device)
            forbidden_teacher_fields = sorted(key for key in batch if key.startswith("teacher_"))
            if forbidden_teacher_fields:
                raise ValueError(f"w/o KD batch contains teacher fields: {forbidden_teacher_fields}")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["images"], batch["text_tokens"], return_dict=True)
            validate_output_contract(outputs, batch["labels"].shape[0], config.feature_dim)
            loss = F.cross_entropy(
                outputs["logits"],
                batch["labels"],
                weight=weights,
                label_smoothing=args.label_smoothing,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.head_parameters()), 1.0)
            optimizer.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_metrics, validation_predictions = evaluate_validation(
            model, validation_loader, device
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        atomic_json_save(history, output_dir / "train_history.json")
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
        payload = checkpoint_payload(model, epoch, validation_metrics, commit)
        weighted_key = (validation_metrics["weighted_f1"], validation_metrics["macro_f1"])
        macro_key = (validation_metrics["macro_f1"], validation_metrics["weighted_f1"])
        if weighted_key > best_weighted:
            best_weighted = weighted_key
            best_epoch = epoch
            best_validation = dict(validation_metrics)
            best_predictions = validation_predictions.copy()
            atomic_torch_save(payload, str(output_dir / "best_weighted_f1.pt"))
        if macro_key > best_macro:
            best_macro = macro_key
            atomic_torch_save(payload, str(output_dir / "best_macro_f1.pt"))
        atomic_torch_save(payload, str(output_dir / "last.pt"))

    if best_validation is None or best_predictions is None or best_epoch is None:
        raise RuntimeError("No validation result was produced")
    save_validation_artifacts(output_dir, best_validation, best_predictions)

    params = parameter_breakdown(model)
    macs = static_head_macs(model, batch_size=1, device=device)
    if int(benchmark["head_params"]) != params["full_head"]:
        raise ValueError(f"Benchmark parameter mismatch for {candidate}")
    if int(benchmark["head_macs"]) != macs["full_head"]:
        raise ValueError(f"Benchmark MAC mismatch for {candidate}")
    result = {
        "candidate": candidate,
        "best_epoch": int(best_epoch),
        "val_accuracy": best_validation["accuracy"],
        "val_weighted_f1": best_validation["weighted_f1"],
        "val_macro_f1": best_validation["macro_f1"],
        "val_precision": best_validation["precision"],
        "val_recall": best_validation["recall"],
        "head_params": params["full_head"],
        "head_macs": macs["full_head"],
        "head_latency_batch1_mean_ms": float(benchmark["mean_ms"]),
        "head_latency_batch1_std_ms": float(benchmark["std_ms"]),
        "peak_memory_mb": float(benchmark["peak_gpu_memory_bytes"]) / (1024.0 * 1024.0),
        "status": "completed",
    }
    atomic_json_save(result, output_dir / "screening_summary.json")
    return result


def mark_pareto(frame: pd.DataFrame) -> pd.Series:
    maximize = ("val_weighted_f1",)
    minimize = ("head_params", "head_latency_batch1_mean_ms")
    flags = []
    for index, row in frame.iterrows():
        dominated = False
        for other_index, other in frame.iterrows():
            if index == other_index:
                continue
            no_worse = all(other[key] >= row[key] for key in maximize) and all(
                other[key] <= row[key] for key in minimize
            )
            strictly_better = any(other[key] > row[key] for key in maximize) or any(
                other[key] < row[key] for key in minimize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=frame.index)


def main() -> None:
    args = parse_args()
    validate_protocol(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_summary(args.benchmark_results, args.candidates)
    rows = [
        train_candidate(args, candidate, device, benchmark[candidate])
        for candidate in args.candidates
    ]
    frame = pd.DataFrame(rows)
    pareto = mark_pareto(frame)
    frame.loc[pareto, "status"] = "pareto"
    frame.loc[~pareto, "status"] = "completed_dominated"
    columns = [
        "candidate",
        "best_epoch",
        "val_accuracy",
        "val_weighted_f1",
        "val_macro_f1",
        "val_precision",
        "val_recall",
        "head_params",
        "head_macs",
        "head_latency_batch1_mean_ms",
        "head_latency_batch1_std_ms",
        "peak_memory_mb",
        "status",
    ]
    frame = frame[columns]
    frame.to_csv(Path(args.output_dir) / "validation_pareto.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
