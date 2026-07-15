#!/usr/bin/env python3
"""Run matched formal w/o-KD and Logits-KD experiments for three seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


SEEDS = [3407, 42, 2024]
CONDITIONS = ("wo_kd", "logits_kd")


def run(command, log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(str(value) for value in command) + "\n")
        handle.flush()
        process = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def config_for(root: Path, condition: str, seed: int, teacher_cache: str):
    output_dir = root / "outputs" / "formal_multiseed" / condition / f"seed_{seed}"
    enable_kd = condition == "logits_kd"
    return {
        "experiment_contract": {
            "matched_pair_seed": seed,
            "condition": condition,
            "only_difference": "Logits KD enabled/disabled",
            "training_protocol": "student_fixed_split_6090_995_950",
        },
        "data": {
            "csv_path": str(root / "data" / "clean" / "task2_clean_consistent.csv"),
            "image_root": str(root / "data" / "CrisisMMD_v2.0"),
            "teacher_cache": teacher_cache if enable_kd else None,
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
            "output_dir": str(output_dir),
            "epochs": 10,
            "batch_size": 8,
            "num_workers": 4,
            "lr": 0.0002,
            "weight_decay": 0.01,
            "disable_kd": not enable_kd,
            "save_best_by": "weighted_f1",
        },
        "loss": {
            "use_class_weight": True,
            "class_weight_method": "inverse_frequency",
            "label_smoothing": 0.05,
        },
        "teacher": {"temperature": 4.0, "confidence_weighted_kd": False},
        "kd_weights": {
            "logits": 0.5 if enable_kd else 0.0,
            "feature": 0.0,
            "gate": 0.0,
            "relation": 0.0,
            "prototype": 0.0,
        },
    }


def evaluate(root: Path, python: str, config_path: Path, output_dir: Path, checkpoint: str, suffix: str, log_path: Path):
    csv_path = root / "data" / "clean" / "task2_clean_consistent.csv"
    image_root = root / "data" / "CrisisMMD_v2.0"
    if suffix:
        target = output_dir / suffix
        target.mkdir(parents=True, exist_ok=True)
    else:
        target = output_dir
    run(
        [
            python, "evaluate.py", "--config", str(config_path), "--csv_path", str(csv_path),
            "--image_root", str(image_root), "--checkpoint", str(output_dir / checkpoint),
            "--output_csv", str(target / "test_predictions.csv"),
            "--metrics_json", str(target / "eval_metrics.json"),
            "--per_class_csv", str(target / "per_class_metrics.csv"),
            "--confusion_csv", str(target / "confusion_matrix.csv"),
            "--batch_size", "8", "--num_workers", "4", "--split", "test",
        ],
        log_path,
        root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    teacher_root = root / "outputs" / "server_mkan_kd_formal"
    cache_report = json.loads((teacher_root / "teacher_cache" / "check_report.json").read_text(encoding="utf-8"))
    if cache_report.get("status") != "PASS":
        raise RuntimeError("Formal teacher cache check_report status is not PASS")
    teacher_cache = str(teacher_root / "teacher_cache" / "mkan_train_logits.pt")
    config_root = root / "configs" / "formal_multiseed"
    log_root = root / "logs" / "formal_multiseed"
    config_root.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for condition in CONDITIONS:
            cfg = config_for(root, condition, seed, teacher_cache)
            output_dir = Path(cfg["train"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_root / f"{condition}_seed_{seed}.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            final_metrics = output_dir / "eval_metrics.json"
            if final_metrics.is_file():
                print(f"skip completed: {condition} seed {seed}")
                continue
            train_log = log_root / f"{condition}_seed_{seed}_train.log"
            run(
                [args.python, "train.py", "--config", str(config_path), "--seed", str(seed)],
                train_log,
                root,
            )
            required = ["best_weighted_f1.pt", "best_macro_f1.pt", "last.pt", "train_history.json", "config_snapshot.json"]
            missing = [name for name in required if not (output_dir / name).is_file()]
            if missing:
                raise RuntimeError(f"Missing training artifacts for {condition} seed {seed}: {missing}")
            evaluate(
                root, args.python, config_path, output_dir, "best_weighted_f1.pt", "",
                log_root / f"{condition}_seed_{seed}_evaluate_weighted.log",
            )
            evaluate(
                root, args.python, config_path, output_dir, "best_macro_f1.pt", "best_macro_sensitivity",
                log_root / f"{condition}_seed_{seed}_evaluate_macro.log",
            )


if __name__ == "__main__":
    main()
