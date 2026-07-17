#!/usr/bin/env python3
"""Summarize staged Logits KD tuning using validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml


def resolve(path: Path) -> tuple[Path, str]:
    marker = path / "reused_complete.json"
    if marker.is_file():
        return Path(json.loads(marker.read_text(encoding="utf-8"))["target"]), "reused_complete"
    return path, "trained"


def trial_row(stage: str, path: Path) -> dict:
    target, provenance = resolve(path)
    payload = torch.load(target / "best_weighted_f1.pt", map_location="cpu", weights_only=False)
    metrics = payload["validation_metrics"]
    args = payload["args"]
    return {
        "stage": stage,
        "trial_dir": str(path),
        "artifact_dir": str(target),
        "provenance": provenance,
        "seed": int(args["seed"]),
        "temperature": float(args["temperature"]),
        "logits_kd_weight": float(args["logits_kd_weight"]),
        **{f"validation_{key}": value for key, value in metrics.items()},
        "test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for stage, directory in (("temperature", root / "temperature"), ("weight", root / "weight")):
        for trial in sorted(path for path in directory.iterdir() if path.is_dir()):
            rows.append(trial_row(stage, trial))
    results = pd.DataFrame(rows).drop_duplicates(subset=["stage", "temperature", "logits_kd_weight"], keep="first")
    results.to_csv(root / "tuning_results.csv", index=False)
    ranking = results.sort_values(["stage", "validation_weighted_f1", "validation_macro_f1"], ascending=[True, False, False]).copy()
    ranking["validation_rank_within_stage"] = ranking.groupby("stage").cumcount() + 1
    ranking.to_csv(root / "tuning_validation_ranking.csv", index=False)
    best_stage_a = ranking[(ranking.stage == "temperature") & (ranking.validation_rank_within_stage == 1)].iloc[0]
    stage_b = ranking[ranking.stage == "weight"]
    best = stage_b.sort_values(["validation_weighted_f1", "validation_macro_f1"], ascending=False).iloc[0]
    best_config = {
        "selection": {"primary": "validation_weighted_f1", "secondary": "validation_macro_f1", "test_used": False, "development_seed": 3407},
        "stage_a_best_temperature": float(best_stage_a.temperature),
        "teacher": {"temperature": float(best.temperature)},
        "kd_weights": {"logits": float(best.logits_kd_weight)},
        "validation_metrics": {"weighted_f1": float(best.validation_weighted_f1), "macro_f1": float(best.validation_macro_f1)},
        "same_as_formal_baseline": bool(float(best.temperature) == 4.0 and float(best.logits_kd_weight) == 0.5),
    }
    (root / "best_config.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")
    lines = [
        "# Staged Logits KD tuning report",
        "",
        "Selection used validation Weighted-F1 first and validation Macro-F1 second on development seed 3407. Test metrics were not loaded or used for hyperparameter selection.",
        "",
        "## Stage A: temperature (weight=0.5)",
        "",
        "| T | Validation Weighted-F1 | Validation Macro-F1 | Provenance |",
        "|---:|---:|---:|---|",
    ]
    for _, row in results[results.stage == "temperature"].sort_values("temperature").iterrows():
        lines.append(f"| {row.temperature:.1f} | {row.validation_weighted_f1:.6f} | {row.validation_macro_f1:.6f} | {row.provenance} |")
    lines.extend(["", f"Selected temperature: **{best_stage_a.temperature:.1f}**.", "", "## Stage B: Logits KD weight", "", "| Weight | Validation Weighted-F1 | Validation Macro-F1 | Provenance |", "|---:|---:|---:|---|"])
    for _, row in results[results.stage == "weight"].sort_values("logits_kd_weight").iterrows():
        lines.append(f"| {row.logits_kd_weight:.2f} | {row.validation_weighted_f1:.6f} | {row.validation_macro_f1:.6f} | {row.provenance} |")
    lines.extend(["", f"Final selected configuration: **T={best.temperature:.1f}, logits_kd_weight={best.logits_kd_weight:.2f}**.", "", "This tuning result is validation-selected and does not replace the original formal three-seed result. If it differs from T=4.0, weight=0.5, it must be evaluated in a separate `formal_multiseed_tuned` directory.", ""])
    (root / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(best_config, indent=2))


if __name__ == "__main__":
    main()
