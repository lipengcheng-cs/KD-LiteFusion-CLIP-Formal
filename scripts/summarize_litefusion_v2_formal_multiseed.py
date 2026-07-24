#!/usr/bin/env python3
"""Summarize validation-only LiteFusion-v2 formal multiseed results."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/litefusion_v2/formal_wo_kd_multiseed"
GROUPED = ROOT / "outputs/litefusion_v2/grouped_gate_optimization"
CANDIDATES = ("v2_c_compact", "v2_g_grouped")
SEEDS = (3407, 42, 2024)
METRICS = ("val_accuracy", "val_weighted_f1", "val_macro_f1", "val_precision", "val_recall")


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return statistics.fmean(values), statistics.stdev(values)


def benchmark_row(rows, candidate, name, mode, batch_size):
    match = [
        row for row in rows
        if row["candidate"] == candidate and row["name"] == name
        and row["mode"] == mode and int(row["batch_size"]) == batch_size
    ]
    if len(match) != 1:
        raise ValueError(f"Missing benchmark row: {candidate}/{name}/{mode}/b{batch_size}")
    return match[0]


def main():
    summaries = {}
    histories = {}
    per_class = {}
    for candidate in CANDIDATES:
        summaries[candidate] = {}
        histories[candidate] = {}
        per_class[candidate] = {}
        for seed in SEEDS:
            directory = FORMAL / candidate / f"seed{seed}"
            if not (directory / "COMPLETED").exists():
                raise FileNotFoundError(f"Incomplete formal run: {directory}")
            summary = load_json(directory / "training_summary.json")
            if summary.get("kd") or summary.get("teacher_cache") is not None:
                raise RuntimeError(f"KD/cache safety failure: {directory}")
            if summary.get("test_evaluation") or summary.get("selection_split") != "val":
                raise RuntimeError(f"Validation-only safety failure: {directory}")
            summaries[candidate][seed] = summary
            histories[candidate][seed] = load_csv(directory / "train_history.csv")
            per_class[candidate][seed] = load_csv(directory / "val_per_class_metrics.csv")

    aggregate_rows = []
    seed_rows = []
    stability_rows = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            summary = summaries[candidate][seed]
            seed_rows.append({
                "candidate": candidate,
                "seed": seed,
                "best_epoch": summary["best_weighted_epoch"],
                **{metric: summary[metric] for metric in METRICS},
            })
            history = histories[candidate][seed]
            last = history[-1]
            stability_rows.append({
                "candidate": candidate,
                "seed": seed,
                "best_epoch": summary["best_weighted_epoch"],
                "first_train_loss": history[0]["train_loss"],
                "last_train_loss": last["train_loss"],
                "best_val_weighted_f1": summary["val_weighted_f1"],
                "last_val_weighted_f1": last["val_weighted_f1"],
                "weighted_f1_best_to_last_drop": float(summary["val_weighted_f1"]) - float(last["val_weighted_f1"]),
                "best_val_macro_f1": summary["val_macro_f1"],
                "last_val_macro_f1": last["val_macro_f1"],
            })
        row = {"candidate": candidate, "n_seeds": len(SEEDS)}
        for metric in METRICS:
            mean, std = mean_std([float(summaries[candidate][seed][metric]) for seed in SEEDS])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        epochs = [int(summaries[candidate][seed]["best_weighted_epoch"]) for seed in SEEDS]
        row["best_epoch_mean"] = statistics.fmean(epochs)
        row["best_epochs"] = "|".join(map(str, epochs))
        aggregate_rows.append(row)

    write_csv(FORMAL / "validation_seed_results.csv", seed_rows)
    write_csv(FORMAL / "validation_multiseed_summary.csv", aggregate_rows)
    write_csv(FORMAL / "training_stability_summary.csv", stability_rows)

    per_class_rows = []
    labels = [row["label"] for row in per_class[CANDIDATES[0]][SEEDS[0]]]
    for candidate in CANDIDATES:
        for label in labels:
            row = {"candidate": candidate, "label": label}
            for metric in ("per_class_precision", "per_class_recall", "per_class_f1"):
                values = [
                    float(next(item for item in per_class[candidate][seed] if item["label"] == label)[metric])
                    for seed in SEEDS
                ]
                mean, std = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            supports = [
                int(next(item for item in per_class[candidate][seed] if item["label"] == label)["support"])
                for seed in SEEDS
            ]
            if len(set(supports)) != 1:
                raise RuntimeError(f"Validation support changed for {label}: {supports}")
            row["support"] = supports[0]
            per_class_rows.append(row)
    write_csv(FORMAL / "validation_per_class_summary.csv", per_class_rows)

    old_rows = load_csv(ROOT / "outputs/litefusion_v2/benchmark_fp32/benchmark_results.csv")
    new_rows = load_csv(GROUPED / "benchmark_after_vectorization_fp32/benchmark_results.csv")
    comparison = []
    for batch in (1, 8):
        for component, name, mode in (
            ("gate", "gate", "component"),
            ("full_head", "full_model", "head_only"),
        ):
            old = benchmark_row(old_rows, "v2_g_grouped", name, mode, batch)
            new = benchmark_row(new_rows, "v2_g_grouped", name, mode, batch)
            comparison.append({
                "component": component,
                "batch_size": batch,
                "before_mean_ms": old["mean_ms"],
                "after_mean_ms": new["mean_ms"],
                "speedup": float(old["mean_ms"]) / float(new["mean_ms"]),
                "after_std_ms": new["std_ms"],
                "after_p50_ms": new["p50_ms"],
                "after_p95_ms": new["p95_ms"],
            })
    write_csv(GROUPED / "optimization_before_after.csv", comparison)
    grouped_payload = {
        "benchmark_fairness_passed": load_json(
            GROUPED / "benchmark_after_vectorization_fp32/benchmark_environment.json"
        )["fairness"]["passed"],
        "head_params": 333509,
        "head_macs": 327808,
        "equivalence": load_json(
            GROUPED / "equivalence_after_vectorization_v2/grouped_gate_equivalence.json"
        ),
        "latency_comparison": comparison,
    }
    summary_path = GROUPED / "optimization_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    summary_path.write_text(json.dumps(grouped_payload, indent=2), encoding="utf-8")

    by_candidate = {row["candidate"]: row for row in aggregate_rows}
    c = by_candidate["v2_c_compact"]
    g = by_candidate["v2_g_grouped"]
    report = f"""# LiteFusion-v2 Final Student Report (Validation Only)

## Safety

All formal runs used frozen OpenAI CLIP, no KD, no teacher cache, and validation-only
checkpoint selection. The test split was not evaluated.

## Three-seed results

| Candidate | Accuracy | Weighted-F1 | Macro-F1 |
|---|---:|---:|---:|
| v2_c_compact | {c['val_accuracy_mean']:.4f} ± {c['val_accuracy_std']:.4f} | {c['val_weighted_f1_mean']:.4f} ± {c['val_weighted_f1_std']:.4f} | {c['val_macro_f1_mean']:.4f} ± {c['val_macro_f1_std']:.4f} |
| v2_g_grouped | {g['val_accuracy_mean']:.4f} ± {g['val_accuracy_std']:.4f} | {g['val_weighted_f1_mean']:.4f} ± {g['val_weighted_f1_std']:.4f} | {g['val_macro_f1_mean']:.4f} ± {g['val_macro_f1_std']:.4f} |

The best Weighted-F1 epochs were {c['best_epochs']} for c_compact and
{g['best_epochs']} for g_grouped. c_compact remains the performance model.
g_grouped is the efficiency model with 333,509 parameters, 327,808 MACs, and
1.2677 ms batch-1 full-head mean latency.

## Training optimization conclusion

C0 (lr=2e-4, no scheduler, inverse-frequency weights, dropout=0.2) remained best.
Cosine scheduling, lr=1e-4, effective-number weights, and dropout=0.3 did not improve
the seed-3407 validation Weighted-F1. The curves show early validation peaks followed
by lower later-epoch scores while train loss decreases, so checkpoint selection is
important.

## Grouped gate conclusion

The original latency came from a Python loop over 32 independent tiny networks,
which launched many small CUDA kernels. The vectorized implementation uses two
grouped 1x1 convolutions and vectorized per-group LayerNorm while preserving dynamic
group gates and the [B, 768] contract. Formal benchmark fairness passed.

Per-seed predictions, per-class metrics, and confusion matrices remain in each seed
directory. Minority-class interpretation must account for the fixed support values
reported in validation_per_class_summary.csv.
"""
    report_path = FORMAL / "FINAL_WO_KD_REPORT.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
