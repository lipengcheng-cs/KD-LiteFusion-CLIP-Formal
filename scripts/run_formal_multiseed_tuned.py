#!/usr/bin/env python3
"""Confirm a validation-selected Logits-KD configuration in a separate three-seed directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch
import yaml


SEEDS = (3407, 42, 2024)
REQUIRED = ("best_weighted_f1.pt", "best_macro_f1.pt", "last.pt", "train_history.json", "config_snapshot.json")


def config_for(root: Path, output: Path, seed: int, temperature: float, weight: float) -> dict:
    return {
        "experiment_contract": {"stage": "formal_multiseed_tuned_confirmation", "seed": seed, "training_protocol": "student_fixed_split_6090_995_950", "hyperparameters_selected_on": "validation_seed_3407_only", "test_used_for_selection": False, "original_formal_results_overwritten": False},
        "data": {"csv_path": str(root / "data" / "clean" / "task2_clean_consistent.csv"), "image_root": str(root / "data" / "CrisisMMD_v2.0"), "teacher_cache": str(root / "outputs" / "server_mkan_kd_formal" / "teacher_cache" / "mkan_train_logits.pt")},
        "model": {"clip_backend": "openai", "clip_model_name": "ViT-L/14@336px", "clip_model_path": "/home/lpc/.cache/clip/ViT-L-14-336px.pt", "clip_frozen": True, "rank": 32, "dropout": 0.2},
        "train": {"output_dir": str(output), "epochs": 10, "batch_size": 8, "num_workers": 0, "lr": 0.0002, "weight_decay": 0.01, "disable_kd": False, "save_best_by": "weighted_f1"},
        "loss": {"use_class_weight": True, "class_weight_method": "inverse_frequency", "label_smoothing": 0.05},
        "teacher": {"temperature": temperature, "confidence_weighted_kd": False},
        "kd_weights": {"logits": weight, "feature": 0.0, "gate": 0.0, "relation": 0.0, "prototype": 0.0},
    }


def complete(path: Path, seed: int, temperature: float, weight: float) -> bool:
    if not all((path / name).is_file() for name in REQUIRED):
        return False
    try:
        snap = json.loads((path / "config_snapshot.json").read_text(encoding="utf-8"))["resolved_args"]
        history = json.loads((path / "train_history.json").read_text(encoding="utf-8"))
        payload = torch.load(path / "best_weighted_f1.pt", map_location="cpu", weights_only=False)
        return int(snap["seed"]) == seed and int(snap["rank"]) == 32 and bool(snap["clip_frozen"]) and abs(float(snap["temperature"]) - temperature) < 1e-12 and abs(float(snap["logits_kd_weight"]) - weight) < 1e-12 and len(history) == 10 and all(torch.isfinite(torch.tensor(float(v))) for v in payload["validation_metrics"].values())
    except Exception:
        return False


def train(root: Path, python: str, path: Path, seed: int, temperature: float, weight: float) -> Path:
    config_root = root / "configs" / "formal_multiseed_tuned"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / f"logits_kd_T{temperature:g}_w{weight:g}_seed_{seed}.yaml"
    config_path.write_text(yaml.safe_dump(config_for(root, path, seed, temperature, weight), sort_keys=False), encoding="utf-8")
    if complete(path, seed, temperature, weight):
        print(f"SKIP_COMPLETE tuned seed={seed}")
        return config_path
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Incomplete tuned directory requires audit: {path}")
    path.mkdir(parents=True, exist_ok=True)
    log = root / "logs" / "formal_multiseed_tuned" / f"seed_{seed}_train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        command = [python, "train.py", "--config", str(config_path), "--seed", str(seed)]
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not complete(path, seed, temperature, weight):
        raise RuntimeError(f"Tuned seed {seed} failed; see {log}")
    return config_path


def evaluate(root: Path, python: str, source: Path, target: Path, config_path: Path, seed: int) -> None:
    if (target / "eval_metrics.json").is_file() and (target / "test_predictions.csv").is_file():
        print(f"SKIP_COMPLETE tuned evaluation seed={seed}")
        return
    target.mkdir(parents=True, exist_ok=True)
    command = [python, "evaluate.py", "--config", str(config_path), "--csv_path", str(root / "data" / "clean" / "task2_clean_consistent.csv"), "--image_root", str(root / "data" / "CrisisMMD_v2.0"), "--checkpoint", str(source / "best_weighted_f1.pt"), "--output_csv", str(target / "test_predictions.csv"), "--metrics_json", str(target / "eval_metrics.json"), "--per_class_csv", str(target / "per_class_metrics.csv"), "--confusion_csv", str(target / "confusion_matrix.csv"), "--batch_size", "8", "--num_workers", "0", "--split", "test"]
    log = root / "logs" / "formal_multiseed_tuned" / f"seed_{seed}_evaluate.log"
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Tuned evaluation failed for seed {seed}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    selected = yaml.safe_load((root / "outputs" / "logits_kd_tuning" / "best_config.yaml").read_text(encoding="utf-8"))
    temperature = float(selected["teacher"]["temperature"])
    weight = float(selected["kd_weights"]["logits"])
    if selected.get("same_as_formal_baseline"):
        print("SKIP_COMPLETE: selected tuning parameters equal original formal baseline")
        return
    output = root / "outputs" / "formal_multiseed_tuned" / "logits_kd"
    tuning_source = root / "outputs" / "logits_kd_tuning" / "weight" / f"T_{str(temperature).replace('.', 'p')}_w_{str(weight).replace('.', 'p')}"
    if not complete(tuning_source, 3407, temperature, weight):
        raise RuntimeError(f"Selected seed-3407 tuning source is incomplete: {tuning_source}")
    seed3407 = output / "seed_3407"
    seed3407.mkdir(parents=True, exist_ok=True)
    marker = {"status": "REUSED_COMPLETE", "source": str(tuning_source), "reason": "Identical validation-selected seed-3407 trial; no retraining", "temperature": temperature, "logits_kd_weight": weight}
    (seed3407 / "reused_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    config3407 = Path(json.loads((tuning_source / "config_snapshot.json").read_text(encoding="utf-8"))["config_path"])
    evaluate(root, args.python, tuning_source, seed3407, config3407, 3407)
    for seed in SEEDS[1:]:
        path = output / f"seed_{seed}"
        config_path = train(root, args.python, path, seed, temperature, weight)
        evaluate(root, args.python, path, path, config_path, seed)
    rows = []
    for seed in SEEDS:
        metrics = json.loads((output / f"seed_{seed}" / "eval_metrics.json").read_text(encoding="utf-8"))
        rows.append({"seed": seed, "temperature": temperature, "logits_kd_weight": weight, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output.parent / "overall_by_seed.csv", index=False)
    summary = {"n_seeds": 3, "temperature": temperature, "logits_kd_weight": weight}
    for metric in ("accuracy", "weighted_f1", "macro_f1", "precision", "recall"):
        summary[f"{metric}_mean"] = float(frame[metric].mean())
        summary[f"{metric}_std"] = float(frame[metric].std(ddof=1))
    (output.parent / "overall_mean_std.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
