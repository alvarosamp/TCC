from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(np.int32, copy=False)
    scores = np.asarray(scores).astype(np.float32, copy=False)

    threshold_info = choose_threshold_by_f1(y_true, scores)
    if threshold is None:
        threshold = threshold_info["threshold"]

    y_pred = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_pr": float(average_precision_score(y_true, scores)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "best_val_threshold": threshold_info,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
