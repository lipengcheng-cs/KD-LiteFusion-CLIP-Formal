#!/usr/bin/env python3
"""Summarize matched formal multiseed student experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SEEDS = [3407, 42, 2024]
CONDITIONS = ["wo_kd", "logits_kd"]
METRICS = ["accuracy", "weighted_f1", "macro_f1", "precision", "recall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    overall_rows, class_rows = [], []
    for condition in CONDITIONS:
        for seed in SEEDS:
            seed_dir = root / condition / f"seed_{seed}"
            metrics = json.loads((seed_dir / "eval_metrics.json").read_text(encoding="utf-8"))
            overall_rows.append({"condition": condition, "seed": seed, **metrics})
            frame = pd.read_csv(seed_dir / "per_class_metrics.csv")
            for row in frame.to_dict("records"):
                class_rows.append({"condition": condition, "seed": seed, **row})
    overall = pd.DataFrame(overall_rows)
    summary_rows = []
    for condition, frame in overall.groupby("condition"):
        row = {"condition": condition, "checkpoint_rule": "best_weighted_f1"}
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(root / "multiseed_overall_mean_std.csv", index=False)
    per_class = pd.DataFrame(class_rows)
    numeric = [column for column in ("precision", "recall", "f1", "support") if column in per_class.columns]
    group_cols = [column for column in ("condition", "class_id", "class_name", "label") if column in per_class.columns]
    class_summary = per_class.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    class_summary.columns = ["_".join(str(part) for part in col if part).rstrip("_") if isinstance(col, tuple) else col for col in class_summary.columns]
    class_summary.to_csv(root / "multiseed_per_class_mean_std.csv", index=False)

    pivot = {row["condition"]: row for row in summary_rows}
    lines = [
        "# Matched multiseed: w/o KD vs Logits KD",
        "",
        "All primary results use each run's `best_weighted_f1.pt`. The best-Macro-F1",
        "checkpoints are preserved only as a separate sensitivity analysis.",
        "",
        "| Condition | Accuracy | Weighted-F1 | Macro-F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = pivot[condition]
        values = [f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}" for metric in METRICS]
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "The only intended matched-pair difference is whether formal-teacher Logits KD",
        "is enabled. Data, split, seed, batch size, epochs, rank, optimizer, learning rate,",
        "class weighting, label smoothing, and checkpoint rules are fixed.",
    ])
    (root / "wo_kd_vs_logits_kd.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
