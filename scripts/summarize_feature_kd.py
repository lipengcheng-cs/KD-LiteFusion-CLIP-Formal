#!/usr/bin/env python3
"""Summarize Feature KD screening and any selected multiseed extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "screening_manifest.json").read_text(encoding="utf-8"))
    results = pd.DataFrame(manifest["trials"])
    results.to_csv(root / "screening_results.csv", index=False)
    lines = ["# Feature KD screening report", "", "Student alignment: final 768-dimensional fused feature after the Reliability-Aware Gate and before the classifier. Teacher alignment: 768-dimensional feature from the formal PASS full cache. Both are converted to FP32, L2-normalized, aligned by `sample_id`, and optimized with mean(1-cosine similarity); teacher features are detached.", "", "Validation/test never read teacher cache. Selection used validation Weighted-F1 then validation Macro-F1 on seed 3407; test results were not used for selection.", "", "| Condition | Feature weight | Validation Weighted-F1 | Validation Macro-F1 | Stop-rule failure |", "|---|---:|---:|---:|---|"]
    for _, row in results.sort_values(["condition", "feature_kd_weight"]).iterrows():
        lines.append(f"| {row.condition} | {row.feature_kd_weight:.2f} | {row.validation_weighted_f1:.6f} | {row.validation_macro_f1:.6f} | {bool(row.failed_stop_rule)} |")
    if manifest["all_failed"]:
        lines.extend(["", "All six configurations failed the predeclared rule; Feature KD stopped before seeds 42 and 2024. Feature KD is not supported for continuation.", ""])
    else:
        selected = yaml.safe_load((root / "selected_feature_config.yaml").read_text(encoding="utf-8"))
        condition = selected["condition"]
        selected_dirs = [Path(next(x["path"] for x in manifest["trials"] if x["condition"] == condition and x["feature_kd_weight"] == selected["feature_kd_weight"]))]
        selected_dirs.extend([root / condition / "selected_seed_42", root / condition / "selected_seed_2024"])
        test_rows = []
        for seed, path in zip((3407, 42, 2024), selected_dirs):
            metrics_path = path / "eval_metrics.json"
            if metrics_path.is_file():
                test_rows.append({"seed": seed, **json.loads(metrics_path.read_text(encoding="utf-8"))})
        lines.extend(["", f"Selected validation configuration: `{selected}`.", ""])
        if len(test_rows) == 3:
            frame = pd.DataFrame(test_rows)
            frame.to_csv(root / "selected_multiseed_test_results.csv", index=False)
            lines.extend(["The selected configuration completed all three seeds. Test means±SD are reported only after validation selection:", "", "| Metric | Mean ± SD |", "|---|---:|"])
            for metric in ("accuracy", "weighted_f1", "macro_f1", "precision", "recall"):
                lines.append(f"| {metric} | {frame[metric].mean():.4f} ± {frame[metric].std(ddof=1):.4f} |")
        lines.append("")
    lines.append("The presence of a full teacher cache alone is not evidence that Feature KD is effective.")
    lines.append("")
    (root / "feature_kd_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "feature_kd_report.md")


if __name__ == "__main__":
    main()
