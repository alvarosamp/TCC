from pathlib import Path
import json
import numpy as np
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENTS_DIR = PROJECT_ROOT / "server" / "data" / "events"
FEEDBACK_DIR = PROJECT_ROOT / "server" / "data" / "feedback"

OUTPUT_DIR = PROJECT_ROOT / "data" / "feedback"
REPORT_DIR = PROJECT_ROOT / "artefacts" / "reports"

OUTPUT_NPZ = OUTPUT_DIR / "feedback_dataset.npz"
REPORT_JSON = REPORT_DIR / "feedback_dataset_report.json"

LABEL_MAP = {
    "confirmed_anomaly": 1,
    "false_negative": 1,
    "confirmed_normal": 0,
    "false_positive": 0,
}

IGNORED_LABELS = {
    "uncertain",
    "discarded",
}

FEATURE_KEYS = [
    "score",
    "threshold",
    "window_size",
    "sampling_rate",
    "mean",
    "std",
    "max",
    "min",
    "free_heap_kb",
    "rssi",
]


def load_json_files(directory: Path):
    if not directory.exists():
        return []

    records = []

    for path in sorted(directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            data["_source_file"] = str(path)
            records.append(data)

        except Exception as exc:
            print(f"Erro ao ler {path}: {exc}")

    return records


def get_event_id(record: dict):
    return record.get("event_id") or record.get("id")


def get_feedback_event_id(record: dict):
    return record.get("event_id")


def extract_features(event: dict):
    features = event.get("features", {}) or {}

    row = {
        "score": event.get("score", 0.0),
        "threshold": event.get("threshold", 0.5),
        "window_size": event.get("window_size", 0),
        "sampling_rate": event.get("sampling_rate", 0),
        "mean": features.get("mean", 0.0),
        "std": features.get("std", 0.0),
        "max": features.get("max", 0.0),
        "min": features.get("min", 0.0),
        "free_heap_kb": features.get("free_heap_kb", 0.0),
        "rssi": features.get("rssi", 0.0),
    }

    return [float(row[key]) for key in FEATURE_KEYS]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    events = load_json_files(EVENTS_DIR)
    feedbacks = load_json_files(FEEDBACK_DIR)

    events_by_id = {}

    for event in events:
        event_id = get_event_id(event)

        if event_id:
            events_by_id[event_id] = event

    X = []
    y = []
    event_ids = []
    labels = []

    ignored_count = 0
    missing_event_count = 0
    invalid_label_count = 0
    label_counts = {}

    for feedback in feedbacks:
        event_id = get_feedback_event_id(feedback)
        label = feedback.get("feedback_label") or feedback.get("label")

        if label is None:
            invalid_label_count += 1
            continue

        label_counts[label] = label_counts.get(label, 0) + 1

        if label in IGNORED_LABELS:
            ignored_count += 1
            continue

        if label not in LABEL_MAP:
            invalid_label_count += 1
            continue

        event = events_by_id.get(event_id)

        if event is None:
            missing_event_count += 1
            continue

        X.append(extract_features(event))
        y.append(LABEL_MAP[label])
        event_ids.append(event_id)
        labels.append(label)

    if len(X) > 0:
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        event_ids = np.array(event_ids, dtype=str)
        labels = np.array(labels, dtype=str)
    else:
        X = np.empty((0, len(FEATURE_KEYS)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        event_ids = np.empty((0,), dtype=str)
        labels = np.empty((0,), dtype=str)

    np.savez(
        OUTPUT_NPZ,
        X=X,
        y=y,
        event_ids=event_ids,
        labels=labels,
        feature_keys=np.array(FEATURE_KEYS, dtype=str),
    )

    class_counts = {
        "normal_0": int((y == 0).sum()),
        "anomaly_1": int((y == 1).sum()),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_dir": str(EVENTS_DIR),
        "feedback_dir": str(FEEDBACK_DIR),
        "output_npz": str(OUTPUT_NPZ),
        "total_events": len(events),
        "total_feedbacks": len(feedbacks),
        "usable_samples": int(len(y)),
        "ignored_feedbacks": int(ignored_count),
        "missing_event_count": int(missing_event_count),
        "invalid_label_count": int(invalid_label_count),
        "label_counts_raw": label_counts,
        "class_counts": class_counts,
        "feature_keys": FEATURE_KEYS,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Feedback dataset construction completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()