"""
Checa prediction drift e performance drift usando eventos coletados pelo servidor.

Prediction drift:
  - compara distribuicao de scores recentes contra uma janela de referencia.
  - mede taxa de anomalias e PSI dos scores.

Performance drift:
  - usa feedback humano quando disponivel.
  - calcula precision/recall/F1 aproximados a partir de false_positive e false_negative.

Entrada padrao:
  server/data/events/*.json
  server/data/feedback/*.json

Saida:
  artefacts/monitoring/prediction_performance_drift_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import math

from server.app.storage import EVENTS_DIR, FEEDBACK_DIR

DEFAULT_OUTPUT = Path("artefacts/monitoring/prediction_performance_drift_report.json")


def load_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for file_path in sorted(path.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return items


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def histogram(values: list[float], bins: list[float]) -> list[int]:
    counts = [0 for _ in range(len(bins) - 1)]
    for value in values:
        for i in range(len(bins) - 1):
            is_last = i == len(bins) - 2
            if bins[i] <= value < bins[i + 1] or (is_last and value == bins[i + 1]):
                counts[i] += 1
                break
    return counts


def psi(expected: list[float], actual: list[float], n_bins: int = 10) -> float:
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    raw_bins = [quantile(expected, i / n_bins) for i in range(n_bins + 1)]
    bins = []
    for value in raw_bins:
        if not bins or value != bins[-1]:
            bins.append(value)
    if len(bins) < 3:
        return 0.0
    expected_counts = histogram(expected, bins)
    actual_counts = histogram(actual, bins)
    expected_total = max(sum(expected_counts), 1)
    actual_total = max(sum(actual_counts), 1)
    eps = 1e-6
    total = 0.0
    for exp_count, act_count in zip(expected_counts, actual_counts):
        exp_pct = max(exp_count / expected_total, eps)
        act_pct = max(act_count / actual_total, eps)
        total += (act_pct - exp_pct) * math.log(act_pct / exp_pct)
    return float(total)

def label_to_binary(label: str | None) -> int | None:
    if label in {"confirmed_anomaly", "false_negative"}:
        return 1
    if label in {"confirmed_normal", "false_positive"}:
        return 0
    return None


def prediction_to_binary(prediction: str | None) -> int | None:
    if prediction == "anomaly":
        return 1
    if prediction == "normal":
        return 0
    return None


def classify_prediction_drift(score_psi: float, anomaly_rate_delta: float) -> str:
    if score_psi >= 0.25 or abs(anomaly_rate_delta) >= 0.20:
        return "high"
    if score_psi >= 0.10 or abs(anomaly_rate_delta) >= 0.10:
        return "moderate"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", type=Path, default=EVENTS_DIR)
    parser.add_argument("--feedback-dir", type=Path, default=FEEDBACK_DIR)
    parser.add_argument("--recent-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    events = load_items(args.events_dir)
    feedback = load_items(args.feedback_dir)
    events = [e for e in events if isinstance(e.get("score"), (int, float))]
    events = sorted(events, key=lambda e: e.get("received_at_utc", ""))

    recent = events[-args.recent_size:]
    reference = events[:-args.recent_size] if len(events) > args.recent_size else []

    recent_scores = [float(e["score"]) for e in recent]
    reference_scores = [float(e["score"]) for e in reference]

    recent_anomaly_rate = mean([1.0 if e.get("prediction") == "anomaly" else 0.0 for e in recent]) if recent else 0.0
    reference_anomaly_rate = mean([1.0 if e.get("prediction") == "anomaly" else 0.0 for e in reference]) if reference else recent_anomaly_rate
    score_psi = psi(reference_scores, recent_scores) if len(reference_scores) else 0.0
    anomaly_rate_delta = recent_anomaly_rate - reference_anomaly_rate
    prediction_drift_level = classify_prediction_drift(score_psi, anomaly_rate_delta)

    by_event = {item.get("event_id"): item for item in feedback}
    y_true: list[int] = []
    y_pred: list[int] = []
    for event in events:
        item = by_event.get(event.get("event_id"))
        if item is None:
            continue
        true_label = label_to_binary(item.get("label"))
        pred_label = prediction_to_binary(event.get("prediction"))
        if true_label is None or pred_label is None:
            continue
        y_true.append(true_label)
        y_pred.append(pred_label)

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None

    performance_level = "unknown"
    if f1 is not None:
        if f1 < 0.65:
            performance_level = "high"
        elif f1 < 0.75:
            performance_level = "moderate"
        else:
            performance_level = "low"

    report = {
        "n_events": len(events),
        "n_reference_events": len(reference),
        "n_recent_events": len(recent),
        "prediction_drift": {
            "level": prediction_drift_level,
            "score_psi": score_psi,
            "reference_anomaly_rate": reference_anomaly_rate,
            "recent_anomaly_rate": recent_anomaly_rate,
            "anomaly_rate_delta": anomaly_rate_delta,
        },
        "performance_drift": {
            "level": performance_level,
            "n_labeled_events": len(y_true),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("PREDICTION/PERFORMANCE DRIFT GERADO")
    print("=" * 80)
    print(f"Eventos: {len(events)}")
    print(f"Prediction drift: {prediction_drift_level}")
    print(f"Performance drift: {performance_level}")
    print(f"Saida: {args.output}")


if __name__ == "__main__":
    main()
