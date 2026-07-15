#!/usr/bin/env python3
"""Select formal teacher weights using validation logits only, then test once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


SEEDS = [3407, 42, 2024]
NATIVE_LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]
LOGIT_COLUMNS = [f"logit_{label}" for label in NATIVE_LABELS]
ORIGINAL_WEIGHTS = np.array([0.18749672, 0.18847652, 0.62402676], dtype=np.float64)
IDENTITY = "mkan_refine_supplied_source_reproduction"
PROTOCOL = "student_fixed_split_6090_995_950"


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
    }


def load_split(root: Path, split: str):
    frames = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / f"{split}_predictions.csv"
        frame = pd.read_csv(path)
        required = {"sample_id", "true_native_id", *LOGIT_COLUMNS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        frame["sample_id"] = frame["sample_id"].astype(str)
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {path}")
        frames.append(frame.sort_values("sample_id").reset_index(drop=True))
    reference_ids = frames[0]["sample_id"].tolist()
    reference_labels = frames[0]["true_native_id"].astype(int).to_numpy()
    for seed, frame in zip(SEEDS[1:], frames[1:]):
        if frame["sample_id"].tolist() != reference_ids:
            raise ValueError(f"{split} sample IDs do not align for seed {seed}")
        if not np.array_equal(frame["true_native_id"].astype(int).to_numpy(), reference_labels):
            raise ValueError(f"{split} labels do not align for seed {seed}")
    logits = np.stack([frame[LOGIT_COLUMNS].to_numpy(dtype=np.float64) for frame in frames])
    return reference_ids, reference_labels, logits


def evaluate_weights(labels, logits, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights < -1e-12).any() or not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError(f"Invalid weights: {weights}")
    ensemble = np.tensordot(weights, logits, axes=(0, 0))
    predictions = ensemble.argmax(axis=1)
    return metrics(labels, predictions), predictions, ensemble


def better(candidate, incumbent):
    return (candidate["weighted_f1"], candidate["macro_f1"]) > (
        incumbent["weighted_f1"], incumbent["macro_f1"]
    )


def simplex_grid(step: float):
    units = int(round(1.0 / step))
    for i in range(units + 1):
        for j in range(units - i + 1):
            k = units - i - j
            yield np.array([i, j, k], dtype=np.float64) / units


def optimized_weights(labels, logits):
    best_metrics = {"weighted_f1": -1.0, "macro_f1": -1.0}
    best_weights = None
    for weights in simplex_grid(0.01):
        result, _, _ = evaluate_weights(labels, logits, weights)
        if better(result, best_metrics):
            best_metrics, best_weights = result, weights.copy()
    center = best_weights
    for w0 in np.arange(max(0.0, center[0] - 0.02), min(1.0, center[0] + 0.02) + 0.0005, 0.001):
        for w1 in np.arange(max(0.0, center[1] - 0.02), min(1.0, center[1] + 0.02) + 0.0005, 0.001):
            w2 = 1.0 - w0 - w1
            if w2 < 0.0 or w2 > 1.0:
                continue
            weights = np.array([w0, w1, w2], dtype=np.float64)
            result, _, _ = evaluate_weights(labels, logits, weights)
            if better(result, best_metrics):
                best_metrics, best_weights = result, weights.copy()
    best_weights = np.clip(best_weights, 0.0, 1.0)
    best_weights /= best_weights.sum()
    return best_weights, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.teacher_root
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    val_ids, val_labels, val_logits = load_split(root, "val")

    candidate_weights = {
        "author_original_weights": ORIGINAL_WEIGHTS / ORIGINAL_WEIGHTS.sum(),
        "equal_weights": np.ones(3, dtype=np.float64) / 3.0,
    }
    single_results = []
    for index, seed in enumerate(SEEDS):
        weights = np.zeros(3, dtype=np.float64)
        weights[index] = 1.0
        result, _, _ = evaluate_weights(val_labels, val_logits, weights)
        single_results.append((result, seed, weights))
    single_results.sort(key=lambda item: (item[0]["weighted_f1"], item[0]["macro_f1"]), reverse=True)
    best_single_metrics, best_single_seed, best_single_weights = single_results[0]
    candidate_weights[f"best_single_seed_{best_single_seed}"] = best_single_weights
    searched_weights, searched_metrics = optimized_weights(val_labels, val_logits)
    candidate_weights["validation_optimized_weights"] = searched_weights

    rows = []
    evaluated = {}
    for name, weights in candidate_weights.items():
        result, _, _ = evaluate_weights(val_labels, val_logits, weights)
        evaluated[name] = {"metrics": result, "weights": weights}
        rows.append({
            "strategy": name,
            "weight_seed_3407": weights[0],
            "weight_seed_42": weights[1],
            "weight_seed_2024": weights[2],
            **result,
        })
    pd.DataFrame(rows).to_csv(report_dir / "ensemble_validation_results.csv", index=False)

    optimized = evaluated["validation_optimized_weights"]
    best_single_name = f"best_single_seed_{best_single_seed}"
    if better(optimized["metrics"], best_single_metrics):
        selected_name = "validation_optimized_weights"
        selection_reason = "Validation-optimized ensemble strictly beats the best single model."
    else:
        selected_name = best_single_name
        selection_reason = (
            "Validation-optimized ensemble does not strictly beat the best single model; "
            "the best single model is selected as required."
        )
    selected = evaluated[selected_name]
    selected_weights = selected["weights"]
    selected_payload = {
        "selected_strategy": selected_name,
        "selected_weights": {str(seed): float(weight) for seed, weight in zip(SEEDS, selected_weights)},
        "selected_validation_metrics": selected["metrics"],
        "selection_primary_metric": "validation_weighted_f1",
        "selection_secondary_metric": "validation_macro_f1",
        "selection_reason": selection_reason,
        "best_single_seed": int(best_single_seed),
        "best_single_validation_metrics": best_single_metrics,
        "test_used_for_selection": False,
        "teacher_identity": IDENTITY,
        "strict_b_spline_reproduction": False,
        "training_protocol": PROTOCOL,
        "checkpoints": [str((root / "checkpoints" / f"ema_seed{seed}.pth").resolve()) for seed in SEEDS],
    }
    (report_dir / "ensemble_selected_weights.json").write_text(
        json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8"
    )

    test_ids, test_labels, test_logits = load_split(root, "test")
    test_metrics, test_predictions, selected_logits = evaluate_weights(
        test_labels, test_logits, selected_weights
    )
    test_payload = {
        **test_metrics,
        "selected_strategy": selected_name,
        "selected_weights": selected_payload["selected_weights"],
        "evaluated_once_after_validation_selection": True,
        "teacher_identity": IDENTITY,
        "strict_b_spline_reproduction": False,
        "training_protocol": PROTOCOL,
    }
    (report_dir / "ensemble_test_metrics.json").write_text(
        json.dumps(test_payload, indent=2) + "\n", encoding="utf-8"
    )
    prediction_frame = pd.DataFrame({
        "sample_id": test_ids,
        "true_native_id": test_labels,
        "pred_native_id": test_predictions,
    })
    for index, label in enumerate(NATIVE_LABELS):
        prediction_frame[f"logit_{label}"] = selected_logits[:, index]
    prediction_frame.to_csv(report_dir / "ensemble_test_predictions.csv", index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_predictions, labels=list(range(5)), zero_division=0
    )
    pd.DataFrame({
        "class_id": list(range(5)),
        "class_name": NATIVE_LABELS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).to_csv(report_dir / "ensemble_per_class_metrics.csv", index=False)
    matrix = confusion_matrix(test_labels, test_predictions, labels=list(range(5)))
    with open(report_dir / "ensemble_confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *NATIVE_LABELS])
        for name, row in zip(NATIVE_LABELS, matrix.tolist()):
            writer.writerow([name, *row])
    print(json.dumps(selected_payload, indent=2))
    print(json.dumps({"test_metrics": test_payload}, indent=2))


if __name__ == "__main__":
    main()
