#!/usr/bin/env python3
"""Paired class-level and bootstrap diagnostics for formal matched KD runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


SEEDS = (3407, 42, 2024)
LABELS = tuple(range(5))
METRICS = ("accuracy", "weighted_f1", "macro_f1")
BOOTSTRAP_SEED = 20260716


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y, pred),
        "weighted_f1": f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
    }


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for label in LABELS:
        indices = np.flatnonzero(y == label)
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(pieces)
    rng.shuffle(result)
    return result


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "lower_95": float(np.quantile(array, 0.025)), "upper_95": float(np.quantile(array, 0.975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    completion = json.loads((root / "completion_report.json").read_text(encoding="utf-8"))
    if completion.get("status") != "PASS":
        raise RuntimeError("Formal completion report must be PASS")
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    pairs: dict[int, pd.DataFrame] = {}
    id_to_name: dict[int, str] = {}
    transitions, class_rows, rescue_regressions, rescue_improvements = [], [], [], []
    mcnemar = {}
    confusion_accumulator = {condition: [] for condition in ("wo_kd", "logits_kd")}
    affected_wide = None
    for seed in SEEDS:
        wo = pd.read_csv(root / "wo_kd" / f"seed_{seed}" / "test_predictions.csv")
        kd = pd.read_csv(root / "logits_kd" / f"seed_{seed}" / "test_predictions.csv")
        pair = wo.merge(kd, on="sample_id", suffixes=("_wo", "_kd"), validate="one_to_one")
        if len(pair) != 950 or not (pair.label_wo == pair.label_kd).all():
            raise RuntimeError(f"Prediction alignment failed for seed {seed}")
        pair["label"] = pair.label_wo.astype(int)
        pairs[seed] = pair
        for _, row in wo[["label", "label_name"]].drop_duplicates().iterrows():
            id_to_name[int(row.label)] = str(row.label_name)
        y = pair.label.to_numpy()
        wo_pred = pair.pred_wo.to_numpy()
        kd_pred = pair.pred_kd.to_numpy()
        wo_correct = wo_pred == y
        kd_correct = kd_pred == y
        b = int((wo_correct & ~kd_correct).sum())
        c = int((~wo_correct & kd_correct).sum())
        mcnemar[str(seed)] = {"wo_correct_kd_wrong": b, "wo_wrong_kd_correct": c, "discordant": b + c, "exact_two_sided_p": exact_mcnemar(b, c)}
        transition = pair.groupby(["pred_wo", "pred_kd"], dropna=False).size().reset_index(name="count")
        for row in transition.to_dict("records"):
            transitions.append({"seed": seed, "wo_prediction_id": row["pred_wo"], "wo_prediction": id_to_name[int(row["pred_wo"])], "logits_kd_prediction_id": row["pred_kd"], "logits_kd_prediction": id_to_name[int(row["pred_kd"])], "count": row["count"]})
        for condition, pred in (("wo_kd", wo_pred), ("logits_kd", kd_pred)):
            precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
            for class_id in LABELS:
                class_rows.append({"condition": condition, "seed": seed, "class_id": class_id, "class_name": id_to_name[class_id], "precision": precision[class_id], "recall": recall[class_id], "f1": f1[class_id], "support": int(support[class_id])})
            matrix = confusion_matrix(y, pred, labels=LABELS, normalize="true")
            confusion_accumulator[condition].append(matrix)
            pd.DataFrame(matrix, index=[id_to_name[x] for x in LABELS], columns=[id_to_name[x] for x in LABELS]).to_csv(output / f"confusion_matrix_{condition}_seed_{seed}.csv")
        affected = pair[pair.label == 0][["sample_id", "label_name_wo", "pred_name_wo", "pred_name_kd"]].copy()
        affected = affected.rename(columns={"label_name_wo": "true_label", "pred_name_wo": f"wo_kd_seed_{seed}", "pred_name_kd": f"logits_kd_seed_{seed}"})
        affected_wide = affected if affected_wide is None else affected_wide.merge(affected.drop(columns="true_label"), on="sample_id", validate="one_to_one")
        rescue = pair[pair.label == 2]
        for target, mask, change in (
            (rescue_regressions, (rescue.pred_wo == rescue.label) & (rescue.pred_kd != rescue.label), "wo_correct_kd_wrong"),
            (rescue_improvements, (rescue.pred_wo != rescue.label) & (rescue.pred_kd == rescue.label), "wo_wrong_kd_correct"),
        ):
            selected = rescue[mask]
            for row in selected.to_dict("records"):
                target.append({"seed": seed, "sample_id": row["sample_id"], "true_label": row["label_name_wo"], "wo_prediction": row["pred_name_wo"], "logits_kd_prediction": row["pred_name_kd"], "change": change})

    pd.DataFrame(transitions).to_csv(output / "prediction_transition_by_seed.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class_metrics_by_seed.csv", index=False)
    class_frame = pd.DataFrame(class_rows)
    class_summary = class_frame.groupby(["condition", "class_id", "class_name"])[["precision", "recall", "f1", "support"]].agg(["mean", "std"]).reset_index()
    class_summary.columns = ["_".join(x for x in col if x) if isinstance(col, tuple) else col for col in class_summary.columns]
    class_summary.to_csv(output / "per_class_mean_std.csv", index=False)
    if affected_wide is None or len(affected_wide) != 7:
        raise RuntimeError(f"Expected exactly 7 affected test samples, found {0 if affected_wide is None else len(affected_wide)}")
    affected_wide.to_csv(output / "affected_individuals_cases.csv", index=False)
    pd.DataFrame(rescue_regressions).to_csv(output / "rescue_regression_cases.csv", index=False)
    pd.DataFrame(rescue_improvements).to_csv(output / "rescue_improvement_cases.csv", index=False)
    (output / "mcnemar_test_by_seed.json").write_text(json.dumps({"test": "exact paired McNemar", "results": mcnemar}, indent=2) + "\n", encoding="utf-8")

    average_rows = []
    for condition, matrices in confusion_accumulator.items():
        average = np.mean(matrices, axis=0)
        for true_id in LABELS:
            for pred_id in LABELS:
                average_rows.append({"condition": condition, "true_class_id": true_id, "true_class": id_to_name[true_id], "predicted_class_id": pred_id, "predicted_class": id_to_name[pred_id], "mean_normalized_count": average[true_id, pred_id]})
    pd.DataFrame(average_rows).to_csv(output / "average_normalized_confusion_matrix.csv", index=False)

    base_y = pairs[SEEDS[0]].label.to_numpy()
    if not all(np.array_equal(base_y, pairs[s].label.to_numpy()) and pairs[SEEDS[0]].sample_id.equals(pairs[s].sample_id) for s in SEEDS):
        raise RuntimeError("Cross-seed sample order/labels do not match")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot: dict[tuple[str, str, str], list[float]] = {}
    for _ in range(args.bootstrap):
        idx = stratified_indices(base_y, rng)
        for condition, column in (("wo_kd", "pred_wo"), ("logits_kd", "pred_kd")):
            per_seed = [scores(base_y[idx], pairs[s][column].to_numpy()[idx]) for s in SEEDS]
            for metric in METRICS:
                boot.setdefault(("three_seed_mean", condition, metric), []).append(float(np.mean([x[metric] for x in per_seed])))
        wo_scores = [scores(base_y[idx], pairs[s].pred_wo.to_numpy()[idx]) for s in SEEDS]
        kd_scores = [scores(base_y[idx], pairs[s].pred_kd.to_numpy()[idx]) for s in SEEDS]
        for metric in METRICS:
            boot.setdefault(("three_seed_mean", "logits_kd_minus_wo_kd", metric), []).append(float(np.mean([kd_scores[i][metric] - wo_scores[i][metric] for i in range(3)])))
        for condition, column in (("wo_kd", "pred_wo"), ("logits_kd", "pred_kd")):
            for class_id in LABELS:
                values = [f1_score(base_y[idx] == class_id, pairs[s][column].to_numpy()[idx] == class_id, zero_division=0) for s in SEEDS]
                boot.setdefault(("three_seed_mean_class_f1", condition, str(class_id)), []).append(float(np.mean(values)))
    ci_rows = []
    for (aggregation, condition, metric), values in boot.items():
        ci = interval(values)
        class_id = int(metric) if aggregation.endswith("class_f1") else None
        ci_rows.append({"aggregation": aggregation, "condition": condition, "metric": "class_f1" if class_id is not None else metric, "class_id": class_id, "class_name": id_to_name[class_id] if class_id is not None else None, "support": int((base_y == class_id).sum()) if class_id is not None else len(base_y), "n_bootstrap": args.bootstrap, "bootstrap_seed": BOOTSTRAP_SEED, **ci})
    ci_payload = {"method": "fixed-seed class-stratified paired bootstrap; shared sample resample across three seeds", "results": ci_rows}
    (output / "bootstrap_confidence_intervals.json").write_text(json.dumps(ci_payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(ci_rows).to_csv(output / "bootstrap_confidence_intervals.csv", index=False)

    paired_ci = pd.DataFrame(ci_rows)
    paired_ci = paired_ci[(paired_ci.condition == "logits_kd_minus_wo_kd") & (paired_ci.aggregation == "three_seed_mean")]
    rescue_regression_counts = pd.DataFrame(rescue_regressions).groupby("seed").size().to_dict() if rescue_regressions else {}
    rescue_improvement_counts = pd.DataFrame(rescue_improvements).groupby("seed").size().to_dict() if rescue_improvements else {}
    lines = ["# Formal KD class-level statistical diagnosis", "", "All analyses use fixed formal test predictions after model selection. These statistics were not used to choose a model or hyperparameter.", "", "## Paired significance", "", "| Seed | w/o correct, KD wrong | w/o wrong, KD correct | Exact McNemar p |", "|---:|---:|---:|---:|"]
    for seed in SEEDS:
        row = mcnemar[str(seed)]
        lines.append(f"| {seed} | {row['wo_correct_kd_wrong']} | {row['wo_wrong_kd_correct']} | {row['exact_two_sided_p']:.6g} |")
    lines.extend(["", "## Three-seed mean paired bootstrap delta", "", "| Metric | Mean | 95% CI |", "|---|---:|---:|"])
    for _, row in paired_ci.iterrows():
        lines.append(f"| {row.metric} | {row['mean']:+.4f} | [{row.lower_95:+.4f}, {row.upper_95:+.4f}] |")
    lines.extend(["", "## Small-support and rescue checks", "", "`affected_individuals` has exactly 7 test samples. Its F1 interval and apparent gains must be interpreted as highly support-sensitive.", "", f"Rescue regressions by seed: {rescue_regression_counts}. Rescue improvements by seed: {rescue_improvement_counts}.", "", "Per-seed class metrics, transition tables, confusion matrices, exact paired tests, and fixed-seed 2,000-replicate stratified bootstrap intervals are stored alongside this report.", ""])
    (output / "class_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
