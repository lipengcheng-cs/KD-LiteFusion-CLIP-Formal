from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    labels: Iterable[int],
    preds: Iterable[int],
    prefix: str = "",
) -> Dict[str, float]:
    y_true = np.asarray(list(labels))
    y_pred = np.asarray(list(preds))
    name = f"{prefix}_" if prefix else ""
    return {
        f"{name}accuracy": accuracy_score(y_true, y_pred),
        f"{name}weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{name}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{name}precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{name}recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def format_metrics(metrics: Dict[str, float], keys: Optional[Iterable[str]] = None) -> str:
    selected = keys or metrics.keys()
    return " | ".join(f"{key}: {metrics[key]:.4f}" for key in selected if key in metrics)


def per_class_metrics_df(labels: Iterable[int], preds: Iterable[int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    y_true = np.asarray(list(labels))
    y_pred = np.asarray(list(preds))
    label_ids = sorted(id_to_label.keys())
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=[id_to_label[i] for i in label_ids],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label_id in label_ids:
        name = id_to_label[label_id]
        values = report.get(name, {})
        rows.append(
            {
                "label_id": label_id,
                "label": name,
                "per_class_precision": values.get("precision", 0.0),
                "per_class_recall": values.get("recall", 0.0),
                "per_class_f1": values.get("f1-score", 0.0),
                "support": int(values.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_df(labels: Iterable[int], preds: Iterable[int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    y_true = np.asarray(list(labels))
    y_pred = np.asarray(list(preds))
    label_ids = sorted(id_to_label.keys())
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    names = [id_to_label[i] for i in label_ids]
    return pd.DataFrame(matrix, index=names, columns=names)
