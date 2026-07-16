#!/usr/bin/env python3
"""Create the publication-facing matched multiseed summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SEEDS = (3407, 42, 2024)
CONDITIONS = ("wo_kd", "logits_kd")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "precision", "recall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    completion = json.loads((root / "completion_report.json").read_text(encoding="utf-8"))
    if completion.get("status") != "PASS":
        raise RuntimeError("completion_report.json is not PASS")
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    overall_rows, class_rows, checkpoint_rows = [], [], []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run = root / condition / f"seed_{seed}"
            metrics = json.loads((run / "eval_metrics.json").read_text(encoding="utf-8"))
            overall_rows.append({"condition": condition, "seed": seed, "checkpoint": "best_weighted_f1.pt", **{m: metrics[m] for m in METRICS}})
            classes = pd.read_csv(run / "per_class_metrics.csv").rename(columns={"label_id": "class_id", "label": "class_name", "per_class_precision": "precision", "per_class_recall": "recall", "per_class_f1": "f1"})
            for row in classes.to_dict("records"):
                class_rows.append({"condition": condition, "seed": seed, "checkpoint": "best_weighted_f1.pt", **row})
            snapshot = json.loads((run / "config_snapshot.json").read_text(encoding="utf-8"))
            checkpoint_rows.append({
                "condition": condition,
                "seed": seed,
                "primary_checkpoint": snapshot.get("primary_checkpoint", "best_weighted_f1.pt"),
                "supplementary_checkpoint": snapshot.get("sensitivity_checkpoint", "best_macro_f1.pt"),
                "main_table_uses": "best_weighted_f1.pt",
                "best_macro_status": "supplementary_only",
            })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(summary_dir / "overall_by_seed.csv", index=False)
    mean_rows = []
    for condition, group in overall.groupby("condition", sort=False):
        row = {"condition": condition, "n_seeds": len(group), "checkpoint": "best_weighted_f1.pt"}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        mean_rows.append(row)
    means = pd.DataFrame(mean_rows)
    means.to_csv(summary_dir / "overall_mean_std.csv", index=False)
    paired = overall.pivot(index="seed", columns="condition", values=list(METRICS))
    paired_rows = []
    for seed in SEEDS:
        row = {"seed": seed, "comparison": "logits_kd_minus_wo_kd"}
        for metric in METRICS:
            row[f"{metric}_wo_kd"] = paired.loc[seed, (metric, "wo_kd")]
            row[f"{metric}_logits_kd"] = paired.loc[seed, (metric, "logits_kd")]
            row[f"{metric}_delta"] = row[f"{metric}_logits_kd"] - row[f"{metric}_wo_kd"]
        paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(summary_dir / "paired_improvement_by_seed.csv", index=False)

    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(summary_dir / "per_class_by_seed.csv", index=False)
    grouped = per_class.groupby(["condition", "class_id", "class_name"], sort=False)
    class_summary = grouped[["precision", "recall", "f1", "support"]].agg(["mean", "std"]).reset_index()
    class_summary.columns = ["_".join(x for x in col if x) if isinstance(col, tuple) else col for col in class_summary.columns]
    class_summary.to_csv(summary_dir / "per_class_mean_std.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(summary_dir / "checkpoint_selection_summary.csv", index=False)

    mean_lookup = {r["condition"]: r for r in mean_rows}
    delta = pd.DataFrame(paired_rows)
    lines = [
        "# Formal matched three-seed summary",
        "",
        "All values are mean ± sample standard deviation over three matched student seeds (3407, 42, 2024). Formal main results use `best_weighted_f1.pt`. Results from `best_macro_f1.pt` are supplementary only and are not mixed into the main table.",
        "",
        "| Condition | Accuracy | Weighted-F1 | Macro-F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = mean_lookup[condition]
        values = [f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}" for m in METRICS]
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired Logits-KD change", "", "| Metric | Mean delta | SD | Positive seeds |", "|---|---:|---:|---:|"])
    for metric in METRICS:
        values = delta[f"{metric}_delta"]
        lines.append(f"| {metric} | {values.mean():+.4f} | {values.std(ddof=1):.4f} | {(values > 0).sum()}/3 |")
    lines.extend([
        "",
        "`affected_individuals` has test support=7. Its per-class F1 is therefore highly unstable; a large single-run change is not evidence of a stable class-level gain.",
        "",
        "Hyperparameter selection and later statistical diagnostics must use validation-only selection rules and must not retroactively select these test results.",
        "",
    ])
    (summary_dir / "formal_multiseed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_dir)


if __name__ == "__main__":
    main()
