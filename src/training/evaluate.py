from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, precision_recall_curve, precision_score, roc_auc_score

# ============================================================
# OBJETIVO DO ARQUIVO
# ============================================================
#
# Este arquivo contem metricas genericas para classificacao binaria
# de anomalias.
#
# Ele nao sabe se o dominio e:
#   - sismico
#   - vibracao
#   - audio
#   - industrial
#
# Ele recebe:
#   y_true
#   scores
#
# E devolve:
#   AUC-PR, AUC-ROC, F1, precision, recall, matriz de confusao
#
# AUC-PR e a metrica primaria porque anomalias sao raras.


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    Escolhe threshold maximizando F1.

    Importante:
      Esse threshold deve ser escolhido na validacao,
      nunca diretamente no teste.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = precision[:-1]
    recall = recall[:-1]
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx])
    }
def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Avalia scores continuos de anomalia.

    scores:
      valores maiores indicam maior chance de anomalia.

    threshold:
      se None, escolhe o melhor threshold por F1 neste proprio conjunto.
      no fluxo correto:
        - escolhe threshold na validacao
        - aplica o mesmo threshold no teste
    """

    y_true = np.asarray(y_true).astype(np.int32, copy=False)
    scores = np.asarray(scores).astype(np.float32, copy=False)

    threshold_info = choose_threshold_by_f1(
        y_true=y_true,
        scores=scores,
    )

    if threshold is None:
        threshold = threshold_info["threshold"]

    y_pred = (scores >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "auc_pr": float(average_precision_score(y_true, scores)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "best_threshold_by_f1": threshold_info,
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_with_validation_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
) -> dict[str, Any]:
    """
    Avaliacao correta para TCC.

    1. Escolhe threshold na validacao.
    2. Usa esse mesmo threshold no teste.

    Isso evita otimizar no conjunto de teste.
    """

    val_metrics = evaluate_scores(
        y_true=y_val,
        scores=scores_val,
        threshold=None,
    )

    threshold = float(val_metrics["threshold"])

    test_metrics = evaluate_scores(
        y_true=y_test,
        scores=scores_test,
        threshold=threshold,
    )

    return {
        "threshold_from_val": threshold,
        "val": val_metrics,
        "test": test_metrics,
    }        