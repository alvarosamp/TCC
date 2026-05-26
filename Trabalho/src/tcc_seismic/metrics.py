from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ThresholdStrategy = Literal["best_f1", "percentile_normal", "target_recall"]


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    strategy: str
    val_metric: float | None = None


def require_sklearn():
    try:
        from sklearn.metrics import (  # type: ignore
            average_precision_score,
            confusion_matrix,
            precision_recall_curve,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from exc
    return {
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
        "precision_recall_curve": precision_recall_curve,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
    }


def choose_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    strategy: ThresholdStrategy = "best_f1",
    normal_percentile: float = 99.0,
    target_recall: float = 0.95,
) -> ThresholdResult:
    """Choose an operational threshold on validation data only."""

    y_val = np.asarray(y_val)
    scores_val = np.asarray(scores_val, dtype=float)

    if strategy == "percentile_normal":
        normal_scores = scores_val[y_val == 0]
        if len(normal_scores) == 0:
            raise ValueError("percentile_normal requires normal validation samples")
        return ThresholdResult(
            threshold=float(np.percentile(normal_scores, normal_percentile)),
            strategy=f"percentile_normal_{normal_percentile:g}",
        )

    metrics = require_sklearn()
    precision, recall, thresholds = metrics["precision_recall_curve"](y_val, scores_val)
    if len(thresholds) == 0:
        raise ValueError("Cannot choose threshold with an empty threshold grid")

    if strategy == "target_recall":
        candidates = np.where(recall[:-1] >= target_recall)[0]
        if len(candidates) == 0:
            idx = int(np.argmax(recall[:-1]))
        else:
            idx = int(candidates[np.argmax(precision[:-1][candidates])])
        return ThresholdResult(
            threshold=float(thresholds[idx]),
            strategy=f"target_recall_{target_recall:g}",
            val_metric=float(recall[idx]),
        )

    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.argmax(f1[:-1]))
    return ThresholdResult(
        threshold=float(thresholds[idx]),
        strategy="best_f1",
        val_metric=float(f1[idx]),
    )


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate scores with a fixed threshold.

    Higher score must mean "more likely event/anomaly".
    """

    metrics = require_sklearn()
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(np.int64)

    precision, recall, f1, _ = metrics["precision_recall_fscore_support"](
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = metrics["confusion_matrix"](y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    out = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_normal": int((y_true == 0).sum()),
        "n_event": int((y_true == 1).sum()),
    }

    if len(np.unique(y_true)) == 2:
        out["auc_pr"] = float(metrics["average_precision_score"](y_true, scores))
        out["auc_roc"] = float(metrics["roc_auc_score"](y_true, scores))
    else:
        out["auc_pr"] = None
        out["auc_roc"] = None

    return out


def evaluate_with_validation_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    strategy: ThresholdStrategy = "best_f1",
) -> dict:
    threshold = choose_threshold(y_val, scores_val, strategy=strategy)
    val = evaluate_scores(y_val, scores_val, threshold.threshold)
    test = evaluate_scores(y_test, scores_test, threshold.threshold)
    return {
        "threshold_strategy": threshold.strategy,
        "threshold_from_validation": threshold.threshold,
        "validation": val,
        "test": test,
    }

