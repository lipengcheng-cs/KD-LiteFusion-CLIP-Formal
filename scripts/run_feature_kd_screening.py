#!/usr/bin/env python3
"""Run guarded seed-3407 Feature KD screening and conditional seed extension."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import yaml


FEATURE_WEIGHTS = (0.05, 0.1, 0.2)
SEEDS = (3407, 42, 2024)
REQUIRED = ("best_weighted_f1.pt", "best_macro_f1.pt", "last.pt", "train_history.json", "config_snapshot.json")


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def validation(path: Path) -> dict:
    payload = torch.load(path / "best_weighted_f1.pt", map_location="cpu", weights_only=False)
    metrics = payload["validation_metrics"]
    if not all(math.isfinite(float(x)) for x in metrics.values()):
        raise RuntimeError(f"Non-finite validation metrics: {path}")
    return metrics


def complete(path: Path, seed: int, feature_weight: float, logits_weight: float, temperature: float) -> bool:
    if not all((path / name).is_file() for name in REQUIRED):
        return False
    try:
        snapshot = json.loads((path / "config_snapshot.json").read_text(encoding="utf-8"))["resolved_args"]
        history = json.loads((path / "train_history.json").read_text(encoding="utf-8"))
        validation(path)
        return (
            int(snapshot["seed"]) == seed
            and int(snapshot["rank"]) == 32
            and bool(snapshot["clip_frozen"])
            and int(snapshot["num_workers"]) == 0
            and abs(float(snapshot["feature_kd_weight"]) - feature_weight) < 1e-12
            and abs(float(snapshot["logits_kd_weight"]) - logits_weight) < 1e-12
            and abs(float(snapshot["temperature"]) - temperature) < 1e-12
            and len(history) == 10
        )
    except Exception:
        return False


def config_for(root: Path, path: Path, condition: str, seed: int, feature_weight: float, logits_weight: float, temperature: float) -> dict:
    return {
        "experiment_contract": {
            "stage": "feature_kd_screening",
            "condition": condition,
            "seed": seed,
            "training_protocol": "student_fixed_split_6090_995_950",
            "student_feature": "post_reliability_aware_gate_pre_classifier_final_768d",
            "teacher_feature": "formal_teacher_full_cache_768d",
            "feature_loss": "mean(1-cosine_similarity(L2(student),L2(detached_teacher)))",
            "selection_primary": "validation_weighted_f1",
            "selection_secondary": "validation_macro_f1",
            "test_used_for_selection": False,
        },
        "data": {
            "csv_path": str(root / "data" / "clean" / "task2_clean_consistent.csv"),
            "image_root": str(root / "data" / "CrisisMMD_v2.0"),
            "teacher_cache": str(root / "outputs" / "server_mkan_kd_formal" / "teacher_cache" / "mkan_train_full.pt"),
        },
        "model": {"clip_backend": "openai", "clip_model_name": "ViT-L/14@336px", "clip_model_path": "/home/lpc/.cache/clip/ViT-L-14-336px.pt", "clip_frozen": True, "rank": 32, "dropout": 0.2},
        "train": {"output_dir": str(path), "epochs": 10, "batch_size": 8, "num_workers": 0, "lr": 0.0002, "weight_decay": 0.01, "disable_kd": False, "save_best_by": "weighted_f1"},
        "loss": {"use_class_weight": True, "class_weight_method": "inverse_frequency", "label_smoothing": 0.05},
        "teacher": {"temperature": temperature, "confidence_weighted_kd": False},
        "kd_weights": {"logits": logits_weight, "feature": feature_weight, "gate": 0.0, "relation": 0.0, "prototype": 0.0},
    }


def run_trial(root: Path, python: str, condition: str, seed: int, feature_weight: float, logits_weight: float, temperature: float, path: Path) -> None:
    config_root = root / "configs" / "feature_kd"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / f"{condition}_seed{seed}_fw{tag(feature_weight)}_T{tag(temperature)}_lw{tag(logits_weight)}.yaml"
    config_path.write_text(yaml.safe_dump(config_for(root, path, condition, seed, feature_weight, logits_weight, temperature), sort_keys=False), encoding="utf-8")
    if complete(path, seed, feature_weight, logits_weight, temperature):
        print(f"SKIP_COMPLETE: {condition} seed={seed} feature_weight={feature_weight}")
        return
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Incomplete Feature-KD directory requires manual audit: {path}")
    path.mkdir(parents=True, exist_ok=True)
    log = root / "logs" / "feature_kd_screening" / f"{condition}_seed{seed}_fw{tag(feature_weight)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        command = [python, "train.py", "--config", str(config_path), "--seed", str(seed)]
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not complete(path, seed, feature_weight, logits_weight, temperature):
        raise RuntimeError(f"Feature-KD trial failed or incomplete; see {log}")


def evaluate(root: Path, python: str, path: Path) -> None:
    if (path / "eval_metrics.json").is_file() and (path / "test_predictions.csv").is_file():
        print(f"SKIP_COMPLETE evaluation: {path}")
        return
    snapshot = json.loads((path / "config_snapshot.json").read_text(encoding="utf-8"))
    config_path = Path(snapshot["config_path"])
    command = [python, "evaluate.py", "--config", str(config_path), "--csv_path", str(root / "data" / "clean" / "task2_clean_consistent.csv"), "--image_root", str(root / "data" / "CrisisMMD_v2.0"), "--checkpoint", str(path / "best_weighted_f1.pt"), "--output_csv", str(path / "test_predictions.csv"), "--metrics_json", str(path / "eval_metrics.json"), "--per_class_csv", str(path / "per_class_metrics.csv"), "--confusion_csv", str(path / "confusion_matrix.csv"), "--batch_size", "8", "--num_workers", "0", "--split", "test"]
    log = root / "logs" / "feature_kd_screening" / f"evaluate_{path.parent.name}_{path.name}.log"
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Selected Feature-KD evaluation failed; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs" / "feature_kd_screening"
    if json.loads((root / "outputs" / "formal_multiseed" / "completion_report.json").read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Formal matched completion must be PASS")
    if json.loads((root / "outputs" / "server_mkan_kd_formal" / "teacher_cache" / "check_report.json").read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Formal full teacher cache must remain PASS")
    formal_summary = pd_read(root / "outputs" / "formal_multiseed" / "summary" / "overall_mean_std.csv")
    wo = formal_summary[formal_summary["condition"] == "wo_kd"].iloc[0]
    kd = formal_summary[formal_summary["condition"] == "logits_kd"].iloc[0]
    if float(kd["weighted_f1_mean"]) + 0.003 < float(wo["weighted_f1_mean"]):
        raise RuntimeError("Formal Logits KD shows overall degradation; Feature KD is blocked")
    best_logits = yaml.safe_load((root / "outputs" / "logits_kd_tuning" / "best_config.yaml").read_text(encoding="utf-8"))
    temperature = float(best_logits["teacher"]["temperature"])
    logits_weight = float(best_logits["kd_weights"]["logits"])
    baseline = root / "outputs" / "formal_multiseed" / "logits_kd" / "seed_3407"
    if temperature != 4.0 or logits_weight != 0.5:
        candidates = list((root / "outputs" / "logits_kd_tuning" / "weight").glob(f"T_{tag(temperature)}_w_{tag(logits_weight)}"))
        if not candidates:
            raise RuntimeError("Selected tuning trial directory not found")
        marker = candidates[0] / "reused_complete.json"
        baseline = Path(json.loads(marker.read_text(encoding="utf-8"))["target"]) if marker.is_file() else candidates[0]
    baseline_metrics = validation(baseline)

    trials = []
    for condition, lw in (("feature_only", 0.0), ("logits_feature", logits_weight)):
        for fw in FEATURE_WEIGHTS:
            path = output / condition / f"seed_3407_fw_{tag(fw)}"
            run_trial(root, args.python, condition, 3407, fw, lw, temperature, path)
            metrics = validation(path)
            failed = metrics["weighted_f1"] < baseline_metrics["weighted_f1"] - 0.003 and metrics["macro_f1"] < baseline_metrics["macro_f1"] - 0.003
            trials.append({"condition": condition, "seed": 3407, "feature_kd_weight": fw, "temperature": temperature, "logits_kd_weight": lw, "validation_weighted_f1": metrics["weighted_f1"], "validation_macro_f1": metrics["macro_f1"], "baseline_validation_weighted_f1": baseline_metrics["weighted_f1"], "baseline_validation_macro_f1": baseline_metrics["macro_f1"], "failed_stop_rule": failed, "path": str(path)})
    viable = [x for x in trials if not x["failed_stop_rule"]]
    manifest = {"selection_uses_test": False, "baseline": str(baseline), "baseline_metrics": baseline_metrics, "trials": trials, "all_failed": not viable}
    output.mkdir(parents=True, exist_ok=True)
    (output / "screening_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not viable:
        print("STOP_ALL_FEATURE_CONFIGS_FAILED: seeds 42 and 2024 will not run")
        return
    best = max(viable, key=lambda row: (row["validation_weighted_f1"], row["validation_macro_f1"]))
    selected = {"condition": best["condition"], "feature_kd_weight": best["feature_kd_weight"], "temperature": best["temperature"], "logits_kd_weight": best["logits_kd_weight"], "selection_primary": "validation_weighted_f1", "selection_secondary": "validation_macro_f1", "test_used": False}
    (output / "selected_feature_config.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    selected_3407 = Path(best["path"])
    evaluate(root, args.python, selected_3407)
    for seed in SEEDS[1:]:
        path = output / best["condition"] / f"selected_seed_{seed}"
        run_trial(root, args.python, best["condition"], seed, best["feature_kd_weight"], best["logits_kd_weight"], best["temperature"], path)
        evaluate(root, args.python, path)
    print("FEATURE_KD_SCREEN_AND_SELECTED_MULTISEED_COMPLETE")


def pd_read(path: Path):
    import pandas as pd

    return pd.read_csv(path)


if __name__ == "__main__":
    main()
