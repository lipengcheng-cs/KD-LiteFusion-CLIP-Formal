#!/usr/bin/env python3
"""Create validation-only LiteFusion-v2 candidate selection artifacts."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/litefusion_v2/student_optimization"
BENCH = ROOT / "outputs/litefusion_v2/grouped_gate_optimization/benchmark_after_vectorization_fp32"


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_rows():
    with (BENCH / "benchmark_results.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latency(rows, candidate, name="full_model", mode="head_only", batch=1):
    match = [
        row for row in rows
        if row["candidate"] == candidate
        and row["name"] == name
        and row["mode"] == mode
        and int(row["batch_size"]) == batch
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one latency row for {candidate}/{name}/{mode}/b{batch}")
    return match[0]


def main():
    fairness = read_json(BENCH / "benchmark_environment.json")["fairness"]
    if not fairness["passed"]:
        raise RuntimeError("Refusing selection because benchmark fairness failed")
    candidates = {
        "v2_c_compact": OUT / "c_compact_seed3407_baseline_10ep",
        "v2_p_precision": OUT / "p_precision_seed3407_baseline_10ep",
        "v2_g_grouped": OUT / "g_grouped_optimized_seed3407_10ep",
    }
    for directory in candidates.values():
        if not (directory / "COMPLETED").exists():
            raise FileNotFoundError(f"Incomplete candidate: {directory}")
    benches = benchmark_rows()
    reasons = {
        "v2_c_compact": "selected: highest validation Weighted-F1 and Macro-F1; primary performance model",
        "v2_p_precision": "capacity ablation: lower validation metrics than c and g, with 1.94x c parameters",
        "v2_g_grouped": "selected: efficiency Pareto model; latency threshold passed and lowest params/MACs",
    }
    selected = {"v2_c_compact", "v2_g_grouped"}
    rows = []
    for candidate, directory in candidates.items():
        summary = read_json(directory / "training_summary.json")
        timing = latency(benches, candidate)
        rows.append({
            "candidate": candidate,
            "config_name": summary["config_name"],
            "best_epoch": summary["best_weighted_epoch"],
            "val_accuracy": summary["val_accuracy"],
            "val_weighted_f1": summary["val_weighted_f1"],
            "val_macro_f1": summary["val_macro_f1"],
            "val_precision": summary["val_precision"],
            "val_recall": summary["val_recall"],
            "head_params": summary["head_params"],
            "head_macs": summary["head_macs"],
            "head_latency_batch1_mean": timing["mean_ms"],
            "head_latency_batch1_std": timing["std_ms"],
            "peak_memory": summary["peak_gpu_memory_mb"],
            "selection_status": "selected" if candidate in selected else "ablation_only",
            "selection_reason": reasons[candidate],
        })
    fields = list(rows[0])
    write_csv(OUT / "final_candidate_selection.csv", rows, fields)

    c_trials = [
        ("C0", "c_compact_seed3407_baseline_10ep"),
        ("C1", "c_compact_C1_cosine_lr2e4"),
        ("C2", "c_compact_C2_cosine_lr1e4"),
        ("C3", "c_compact_C3_effective_num"),
        ("C4", "c_compact_C4_dropout03"),
    ]
    trial_rows = []
    for trial, directory_name in c_trials:
        summary = read_json(OUT / directory_name / "training_summary.json")
        trial_rows.append({
            "trial": trial,
            "config_name": directory_name,
            "best_epoch": summary["best_weighted_epoch"],
            "val_accuracy": summary["val_accuracy"],
            "val_weighted_f1": summary["val_weighted_f1"],
            "val_macro_f1": summary["val_macro_f1"],
            "scheduler": summary["scheduler"],
            "lr": summary["lr"],
            "class_weight_method": summary["class_weight_method"],
            "dropout": summary["dropout"],
        })
    write_csv(OUT / "c_training_optimization_comparison.csv", trial_rows, list(trial_rows[0]))

    report = f"""# LiteFusion-v2 Student Selection

Selection used validation metrics and the fair FP32 benchmark only. No test evaluation,
knowledge distillation, or teacher cache was used.

## Decision

- **v2_c_compact** is the primary performance model: Weighted-F1
  {rows[0]['val_weighted_f1']:.6f}, Macro-F1 {rows[0]['val_macro_f1']:.6f}.
- **v2_g_grouped** is the efficiency Pareto model: Weighted-F1
  {rows[2]['val_weighted_f1']:.6f}, {rows[2]['head_params']:,} parameters,
  batch-1 full-head latency {float(rows[2]['head_latency_batch1_mean']):.4f} ms.
- **v2_p_precision** remains a capacity ablation because it is larger and has lower
  validation metrics than both selected models.

## Fixed formal training configuration

seed-specific runs use 10 epochs, AdamW, lr=2e-4, no scheduler, weight decay=0.01,
inverse-frequency class weights, label smoothing=0.05, dropout=0.2, frozen OpenAI
CLIP, validation-only checkpoint selection, and no KD.

Benchmark fairness passed: {fairness['passed']}.
"""
    report_path = OUT / "SELECTION_REPORT.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
