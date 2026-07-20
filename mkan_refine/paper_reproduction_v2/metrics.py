"""Metrics for the fixed five-class MKAN paper protocol."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(labels, predictions) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "per_class_f1": f1_score(labels, predictions, average=None, labels=list(range(5)), zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(5))).tolist(),
        "sample_count": int(labels.size),
    }


def selection_key(metrics: dict) -> tuple[float, float, float]:
    return (float(metrics["weighted_f1"]), float(metrics["accuracy"]), float(metrics["macro_f1"]))
