#!/usr/bin/env python3
"""Strictly validate the six matched formal student experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch


SEEDS = (3407, 42, 2024)
CONDITIONS = ("wo_kd", "logits_kd")
EXPECTED_SPLITS = {"train": 6090, "val": 995, "test": 950}
REQUIRED = (
    "best_weighted_f1.pt",
    "best_macro_f1.pt",
    "last.pt",
    "eval_metrics.json",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "test_predictions.csv",
    "train_history.json",
)


def finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def add(checks: list[dict], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": str(detail)})


def load_snapshot(run_dir: Path) -> tuple[Path | None, dict]:
    for name in ("config_snapshot.yaml", "config_snapshot.yml", "config_snapshot.json"):
        path = run_dir / name
        if path.is_file():
            if path.suffix == ".json":
                return path, json.loads(path.read_text(encoding="utf-8"))
            import yaml

            return path, yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return None, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output_root = (args.output_root or project / "outputs" / "formal_multiseed").resolve()
    teacher_root = project / "outputs" / "server_mkan_kd_formal" / "teacher_cache"
    cache_report_path = teacher_root / "check_report.json"
    formal_cache = teacher_root / "mkan_train_logits.pt"

    global_checks: list[dict] = []
    csv_path = project / "data" / "clean" / "task2_clean_consistent.csv"
    frame = pd.read_csv(csv_path)
    split_column = "split"
    counts = frame[split_column].astype(str).str.lower().value_counts().to_dict()
    add(global_checks, "fixed_split_counts", all(counts.get(k, 0) == v for k, v in EXPECTED_SPLITS.items()), counts)
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8")) if cache_report_path.is_file() else {}
    add(global_checks, "formal_teacher_cache_report", cache_report.get("status") == "PASS", cache_report_path)
    add(global_checks, "formal_logits_cache_exists", formal_cache.is_file(), formal_cache)

    runs: list[dict] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_dir = output_root / condition / f"seed_{seed}"
            checks: list[dict] = []
            missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
            snapshot_path, snapshot = load_snapshot(run_dir) if run_dir.is_dir() else (None, {})
            if snapshot_path is None:
                missing.append("config_snapshot.yaml|json")
            add(checks, "required_artifacts", not missing, missing or "all present")
            resolved = snapshot.get("resolved_args", snapshot.get("args", {}))
            config = snapshot.get("config", snapshot)
            contract = config.get("experiment_contract", {})
            model = config.get("model", {})
            data = config.get("data", {})
            train = config.get("train", {})
            weights = config.get("kd_weights", {})
            configured_seed = resolved.get("seed", contract.get("matched_pair_seed"))
            add(checks, "seed", configured_seed == seed and contract.get("matched_pair_seed") == seed, configured_seed)
            add(checks, "training_protocol", contract.get("training_protocol") == "student_fixed_split_6090_995_950", contract.get("training_protocol"))
            add(checks, "rank_32", resolved.get("rank", model.get("rank")) == 32, resolved.get("rank", model.get("rank")))
            add(checks, "clip_frozen", bool(resolved.get("clip_frozen", model.get("clip_frozen"))), resolved.get("clip_frozen", model.get("clip_frozen")))
            cache = resolved.get("teacher_cache", data.get("teacher_cache"))
            disable_kd = bool(resolved.get("disable_kd", train.get("disable_kd")))
            logits_weight = float(resolved.get("logits_kd_weight", weights.get("logits", 0.0)) or 0.0)
            if condition == "logits_kd":
                add(checks, "formal_teacher_cache_used", (not disable_kd) and logits_weight > 0 and Path(str(cache)).resolve() == formal_cache.resolve() and cache_report.get("status") == "PASS", cache)
            else:
                add(checks, "teacher_cache_not_used", disable_kd and cache in (None, "") and logits_weight == 0.0, cache)

            for name in ("eval_metrics.json", "train_history.json"):
                path = run_dir / name
                if path.is_file():
                    try:
                        add(checks, f"finite_{name}", finite(json.loads(path.read_text(encoding="utf-8"))), path)
                    except Exception as exc:
                        add(checks, f"finite_{name}", False, exc)
            for name in ("per_class_metrics.csv", "confusion_matrix.csv", "test_predictions.csv"):
                path = run_dir / name
                if path.is_file():
                    try:
                        table = pd.read_csv(path)
                        numeric = table.select_dtypes(include="number")
                        add(checks, f"finite_{name}", not numeric.isin([float("inf"), float("-inf")]).any().any() and not numeric.isna().any().any(), path)
                        if name == "test_predictions.csv":
                            add(checks, "test_prediction_count", len(table) == EXPECTED_SPLITS["test"], len(table))
                    except Exception as exc:
                        add(checks, f"finite_{name}", False, exc)
            for name in ("best_weighted_f1.pt", "best_macro_f1.pt", "last.pt"):
                path = run_dir / name
                if not path.is_file():
                    continue
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    ck_args = payload.get("args", {})
                    add(checks, f"{name}_seed", int(ck_args.get("seed", -1)) == seed, ck_args.get("seed"))
                    add(checks, f"{name}_rank", int(ck_args.get("rank", -1)) == 32, ck_args.get("rank"))
                    add(checks, f"{name}_clip_frozen", bool(ck_args.get("clip_frozen")), ck_args.get("clip_frozen"))
                    add(checks, f"{name}_finite", finite(payload), "all checkpoint tensors/metrics finite")
                except Exception as exc:
                    add(checks, f"{name}_readable", False, exc)
            status = "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL"
            runs.append({"condition": condition, "seed": seed, "run_dir": str(run_dir), "status": status, "checks": checks})

    status = "PASS" if all(x["status"] == "PASS" for x in global_checks) and all(x["status"] == "PASS" for x in runs) else "FAIL"
    report = {"status": status, "expected_split_counts": EXPECTED_SPLITS, "global_checks": global_checks, "runs": runs}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "completion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Formal matched multiseed completion report", "", f"Overall status: **{status}**", "", "Expected fixed split: train=6090, val=995, test=950.", "", "| Condition | Seed | Status | Failed checks |", "|---|---:|---|---|"]
    for run in runs:
        failed = [c["name"] for c in run["checks"] if c["status"] != "PASS"]
        lines.append(f"| {run['condition']} | {run['seed']} | {run['status']} | {', '.join(failed) or '-'} |")
    lines.extend(["", "Formal-teacher Logits KD is accepted only when the formal cache check report is PASS. The w/o-KD runs must not load any teacher cache.", ""])
    (output_root / "completion_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(status)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
