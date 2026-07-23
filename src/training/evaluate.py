from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_auc_score


# ============================================================
# MÉTRICAS ORIENTADAS A EVENTOS E SÉRIES TEMPORAIS
# ============================================================

def _get_anomaly_segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Retorna lista de (start, end) dos segmentos contíguos de anomalia."""
    y = np.asarray(y, dtype=np.int32)
    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(y) - 1))
    return segments


def point_adjust_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Point-Adjust: se qualquer ponto de um segmento anômalo for detectado,
    todos os pontos daquele segmento são considerados detectados (TP).

    Referência: Xu et al. (2018) DONUT; padrão em benchmarks de anomalia.
    """
    y_pred_pa = y_pred.copy()
    for start, end in _get_anomaly_segments(y_true):
        if y_pred[start:end + 1].any():
            y_pred_pa[start:end + 1] = 1
    return y_pred_pa


def pa_f1_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    PA-F1: F1 com point-adjust.

    Mais justo para anomalias em janelas com sobreposição, pois não penaliza
    detecção de onset quando o resto do evento é capturado.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = (np.asarray(scores) >= threshold).astype(np.int32)
    y_pred_pa = point_adjust_predictions(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_pa, labels=[0, 1]).ravel()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"pa_f1": float(f1), "pa_precision": float(prec), "pa_recall": float(rec)}


def event_f1_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    overlap_threshold: float = 0.1,
) -> dict[str, float]:
    """
    Event-F1: F1 no nível de segmento (evento), não de janela.

    Um evento verdadeiro é detectado (TP) se ≥ overlap_threshold de suas janelas
    foram preditas como anômala. Um alarme sem evento correspondente é FP.

    Referência: Tatbul et al. (2018) Precision and Recall for Time Series.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = (np.asarray(scores) >= threshold).astype(np.int32)
    true_segs = _get_anomaly_segments(y_true)
    pred_segs = _get_anomaly_segments(y_pred)

    if not true_segs:
        fp = len(pred_segs)
        return {"event_f1": 0.0, "event_precision": 0.0, "event_recall": 0.0,
                "event_tp": 0, "event_fp": fp, "event_fn": 0}

    matched_true = set()
    matched_pred = set()

    for pi, (ps, pe) in enumerate(pred_segs):
        for ti, (ts, te) in enumerate(true_segs):
            overlap = max(0, min(pe, te) - max(ps, ts) + 1)
            true_len = te - ts + 1
            if overlap / true_len >= overlap_threshold:
                matched_true.add(ti)
                matched_pred.add(pi)

    tp = len(matched_true)
    fn = len(true_segs) - tp
    fp = len(pred_segs) - len(matched_pred)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        "event_f1": float(f1),
        "event_precision": float(prec),
        "event_recall": float(rec),
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
    }


def vus_pr_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    buffer_sizes: list[int] | None = None,
) -> dict[str, float]:
    """
    VUS-PR (Volume Under the Surface — Precision-Recall).

    Computa AUC-PR para múltiplos tamanhos de buffer de predição e integra.
    Buffer=0 equivale ao AUC-PR padrão. Buffers maiores tornam a avaliação
    mais tolerante a desalinhamentos temporais.

    Referência: Kim et al. (2022) "Towards a Rigorous Evaluation of
    Time-Series Anomaly Detection". AAAI 2022.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    if buffer_sizes is None:
        # buffers de 0 a 5% do comprimento da série, máx 20 janelas
        max_buf = min(20, max(1, len(y_true) // 20))
        buffer_sizes = list(range(0, max_buf + 1))

    auc_prs: list[float] = []
    for buf in buffer_sizes:
        if buf == 0:
            y_buf = y_true
        else:
            # dilata os segmentos de anomalia pelo buffer
            y_buf = y_true.copy()
            for start, end in _get_anomaly_segments(y_true):
                lo = max(0, start - buf)
                hi = min(len(y_true) - 1, end + buf)
                y_buf[lo:hi + 1] = 1

        if y_buf.sum() == 0 or y_buf.sum() == len(y_buf):
            auc_prs.append(float("nan"))
            continue
        auc_prs.append(float(average_precision_score(y_buf, scores)))

    valid = [v for v in auc_prs if not np.isnan(v)]
    vus_pr = float(np.mean(valid)) if valid else float("nan")
    return {
        "vus_pr": vus_pr,
        "vus_pr_per_buffer": {str(b): v for b, v in zip(buffer_sizes, auc_prs)},
    }

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

    threshold_info = choose_threshold(
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

    auc_pr = float(average_precision_score(y_true, scores))

    try:
        auc_roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc_roc = float("nan")

    pa = pa_f1_score(y_true, scores, threshold)
    ev = event_f1_score(y_true, scores, threshold)
    vus = vus_pr_score(y_true, scores)

    return {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "threshold": float(threshold),
        "best_threshold_by_f1": threshold_info,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        # métricas orientadas a eventos / séries temporais
        "pa_f1": pa["pa_f1"],
        "pa_precision": pa["pa_precision"],
        "pa_recall": pa["pa_recall"],
        "event_f1": ev["event_f1"],
        "event_precision": ev["event_precision"],
        "event_recall": ev["event_recall"],
        "event_tp": ev["event_tp"],
        "event_fp": ev["event_fp"],
        "event_fn": ev["event_fn"],
        "vus_pr": vus["vus_pr"],
        "vus_pr_per_buffer": vus["vus_pr_per_buffer"],
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