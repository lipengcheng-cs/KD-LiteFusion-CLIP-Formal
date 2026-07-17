#!/usr/bin/env python3
"""Run validation-only staged Logits KD tuning without a 3x3 grid."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import yaml


DEV_SEED = 3407
TEMPERATURES = (2.0, 4.0, 6.0)
WEIGHTS = (0.25, 0.5, 1.0)
BASELINE = (4.0, 0.5)
REQUIRED = ("best_weighted_f1.pt", "best_macro_f1.pt", "last.pt", "train_history.json", "config_snapshot.json")


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def config_for(root: Path, output: Path, temperature: float, weight: float) -> dict:
    return {
        "experiment_contract": {
            "stage": "validation_only_logits_kd_tuning",
            "development_seed": DEV_SEED,
            "training_protocol": "student_fixed_split_6090_995_950",
            "selection_primary": "validation_weighted_f1",
            "selection_secondary": "validation_macro_f1",
            "test_used_for_selection": False,
        },
        "data": {
            "csv_path": str(root / "data" / "clean" / "task2_clean_consistent.csv"),
            "image_root": str(root / "data" / "CrisisMMD_v2.0"),
            "teacher_cache": str(root / "outputs" / "server_mkan_kd_formal" / "teacher_cache" / "mkan_train_logits.pt"),
        },
        "model": {
            "clip_backend": "openai",
            "clip_model_name": "ViT-L/14@336px",
            "clip_model_path": "/home/lpc/.cache/clip/ViT-L-14-336px.pt",
            "clip_frozen": True,
            "rank": 32,
            "dropout": 0.2,
        },
        "train": {
            "output_dir": str(output),
            "epochs": 10,
            "batch_size": 8,
            "num_workers": 0,
            "lr": 0.0002,
            "weight_decay": 0.01,
            "disable_kd": False,
            "save_best_by": "weighted_f1",
        },
        "loss": {"use_class_weight": True, "class_weight_method": "inverse_frequency", "label_smoothing": 0.05},
        "teacher": {"temperature": float(temperature), "confidence_weighted_kd": False},
        "kd_weights": {"logits": float(weight), "feature": 0.0, "gate": 0.0, "relation": 0.0, "prototype": 0.0},
    }


def metrics_from_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload["validation_metrics"]
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError(f"Non-finite validation metric in {path}")
    return metrics


def complete(directory: Path, temperature: float, weight: float) -> bool:
    if not all((directory / name).is_file() for name in REQUIRED):
        return False
    try:
        snapshot = json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
        resolved = snapshot["resolved_args"]
        history = json.loads((directory / "train_history.json").read_text(encoding="utf-8"))
        metrics_from_checkpoint(directory / "best_weighted_f1.pt")
        return (
            int(resolved["seed"]) == DEV_SEED
            and int(resolved["rank"]) == 32
            and bool(resolved["clip_frozen"])
            and int(resolved["num_workers"]) == 0
            and abs(float(resolved["temperature"]) - temperature) < 1e-12
            and abs(float(resolved["logits_kd_weight"]) - weight) < 1e-12
            and len(history) == int(resolved["epochs"]) == 10
        )
    except Exception:
        return False


def reference(directory: Path, target: Path, temperature: float, weight: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "REUSED_COMPLETE",
        "target": str(target.resolve()),
        "temperature": temperature,
        "logits_kd_weight": weight,
        "seed": DEV_SEED,
        "reason": "An identical formal matched run is already complete; no retraining.",
    }
    (directory / "reused_complete.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve(directory: Path) -> Path:
    marker = directory / "reused_complete.json"
    if marker.is_file():
        return Path(json.loads(marker.read_text(encoding="utf-8"))["target"])
    return directory


def run_trial(root: Path, python: str, config_root: Path, directory: Path, stage: str, temperature: float, weight: float, reuse: Path | None = None) -> None:
    cfg = config_for(root, directory, temperature, weight)
    config_path = config_root / f"{stage}_T{tag(temperature)}_w{tag(weight)}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if reuse is not None:
        reference(directory, reuse, temperature, weight)
        print(f"SKIP_COMPLETE reuse: {stage} T={temperature} weight={weight} -> {reuse}")
        return
    if complete(directory, temperature, weight):
        print(f"SKIP_COMPLETE: {stage} T={temperature} weight={weight}")
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Incomplete tuning directory requires manual audit before recovery: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    log = root / "logs" / "logits_kd_tuning" / f"{stage}_T{tag(temperature)}_w{tag(weight)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        command = [python, "train.py", "--config", str(config_path), "--seed", str(DEV_SEED)]
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not complete(directory, temperature, weight):
        raise RuntimeError(f"Tuning trial failed or incomplete; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs" / "logits_kd_tuning"
    config_root = root / "configs" / "logits_kd_tuning"
    report = json.loads((root / "outputs" / "formal_multiseed" / "completion_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("Formal matched completion report must be PASS")
    cache_report = json.loads((root / "outputs" / "server_mkan_kd_formal" / "teacher_cache" / "check_report.json").read_text(encoding="utf-8"))
    if cache_report.get("status") != "PASS":
        raise RuntimeError("Formal teacher cache must remain PASS")

    baseline = root / "outputs" / "formal_multiseed" / "logits_kd" / f"seed_{DEV_SEED}"
    if not (baseline / "best_weighted_f1.pt").is_file():
        raise RuntimeError("Formal T=4, weight=0.5 baseline is missing")
    stage_a = []
    for temperature in TEMPERATURES:
        directory = output / "temperature" / f"T_{tag(temperature)}_w_0p5"
        run_trial(root, args.python, config_root, directory, "temperature", temperature, 0.5, baseline if temperature == BASELINE[0] else None)
        metrics = metrics_from_checkpoint(resolve(directory) / "best_weighted_f1.pt")
        stage_a.append((metrics["weighted_f1"], metrics["macro_f1"], -abs(temperature - 4.0), temperature, directory))
    best_temperature = max(stage_a)[3]
    (output / "stage_a_selection.json").write_text(json.dumps({"best_temperature": best_temperature, "rule": ["validation_weighted_f1", "validation_macro_f1"], "test_used": False}, indent=2) + "\n", encoding="utf-8")

    stage_a_source = output / "temperature" / f"T_{tag(best_temperature)}_w_0p5"
    stage_a_target = resolve(stage_a_source)
    for weight in WEIGHTS:
        directory = output / "weight" / f"T_{tag(best_temperature)}_w_{tag(weight)}"
        reuse = stage_a_target if weight == 0.5 else None
        run_trial(root, args.python, config_root, directory, "weight", best_temperature, weight, reuse)
    print(f"STAGED_TUNING_COMPLETE best_stage_a_temperature={best_temperature}")


if __name__ == "__main__":
    main()
