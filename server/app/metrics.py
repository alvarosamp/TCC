"""Endpoint Prometheus em texto puro, sem dependencia externa."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from server.app.storage import DEVICES_DIR, EVENTS_DIR, FEEDBACK_DIR, OTA_REPORTS_DIR, list_json_files


def _prediction_counts(events: list[dict[str, Any]], label: str) -> int:
    return sum(1 for event in events if event.get("prediction") == label)


def _ota_status_count(reports: list[dict[str, Any]], status: str) -> int:
    return sum(1 for report in reports if report.get("status") == status)


def render_prometheus_metrics() -> str:
    devices = [d for d in list_json_files(DEVICES_DIR) if d.get("device_id")]
    events = list_json_files(EVENTS_DIR)
    feedback = list_json_files(FEEDBACK_DIR)
    ota_reports = list_json_files(OTA_REPORTS_DIR)

    lines = [
        "# HELP tcc_devices_total Total de dispositivos registrados.",
        "# TYPE tcc_devices_total gauge",
        f"tcc_devices_total {len(devices)}",
        "# HELP tcc_events_total Total de eventos recebidos.",
        "# TYPE tcc_events_total counter",
        f"tcc_events_total {len(events)}",
        "# HELP tcc_feedback_total Total de feedbacks humanos recebidos.",
        "# TYPE tcc_feedback_total counter",
        f"tcc_feedback_total {len(feedback)}",
        "# HELP tcc_ota_reports_total Total de reports OTA por status.",
        "# TYPE tcc_ota_reports_total counter",
        f'tcc_ota_reports_total{{status="success"}} {_ota_status_count(ota_reports, "success")}',
        f'tcc_ota_reports_total{{status="failed"}} {_ota_status_count(ota_reports, "failed")}',
        "# HELP tcc_predictions_total Total de predicoes por classe.",
        "# TYPE tcc_predictions_total counter",
        f'tcc_predictions_total{{prediction="normal"}} {_prediction_counts(events, "normal")}',
        f'tcc_predictions_total{{prediction="anomaly"}} {_prediction_counts(events, "anomaly")}',
        f'tcc_predictions_total{{prediction="uncertain"}} {_prediction_counts(events, "uncertain")}',
    ]

    if events:
        avg_score = sum(float(e.get("score", 0.0)) for e in events) / len(events)
        lines.extend([
            "# HELP tcc_score_average Media dos scores recebidos.",
            "# TYPE tcc_score_average gauge",
            f"tcc_score_average {avg_score}",
        ])

    newline = chr(10)
    return newline.join(lines) + newline
