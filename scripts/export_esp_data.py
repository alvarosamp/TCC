from pathlib import Path
import json
import csv


BASE_DIR = Path(__file__).resolve().parents[1]

POSSIBLE_DATA_DIRS = [
    BASE_DIR / "server" / "app" / "data",
    BASE_DIR / "data",
    BASE_DIR / "storage",
]

EXPORT_DIR = BASE_DIR / "data" / "exported"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_existing_dir(possible_dirs: list[Path], name: str) -> Path | None:
    for base_dir in possible_dirs:
        candidate = base_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def find_events_dir() -> Path | None:
    possible_names = [
        "events",
        "EVENTS",
    ]

    for base_dir in POSSIBLE_DATA_DIRS:
        for name in possible_names:
            candidate = base_dir / name
            if candidate.exists() and candidate.is_dir():
                return candidate

    matches = list(BASE_DIR.rglob("events"))
    for match in matches:
        if match.is_dir():
            return match

    return None


def find_devices_dir() -> Path | None:
    possible_names = [
        "devices",
        "DEVICES",
    ]

    for base_dir in POSSIBLE_DATA_DIRS:
        for name in possible_names:
            candidate = base_dir / name
            if candidate.exists() and candidate.is_dir():
                return candidate

    matches = list(BASE_DIR.rglob("devices"))
    for match in matches:
        if match.is_dir():
            return match

    return None


def find_status_history_dir(devices_dir: Path | None) -> Path | None:
    if devices_dir is not None:
        possible = [
            devices_dir / "_status_history",
            devices_dir / "status_history",
            devices_dir / "history",
        ]

        for candidate in possible:
            if candidate.exists() and candidate.is_dir():
                return candidate

    matches = list(BASE_DIR.rglob("_status_history"))
    for match in matches:
        if match.is_dir():
            return match

    matches = list(BASE_DIR.rglob("status_history"))
    for match in matches:
        if match.is_dir():
            return match

    return None


def load_json_file(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as error:
        print(f"Erro ao ler {path}: {error}")
        return None


def export_csv(rows: list[dict], output_path: Path):
    if not rows:
        print(f"Nenhum dado para exportar em {output_path.name}")
        return

    columns = sorted({key for row in rows for key in row.keys()})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exportado: {output_path}")
    print(f"Total de linhas: {len(rows)}")


def flatten_event(event: dict) -> dict:
    features = event.get("features") or {}
    extra = event.get("extra") or {}

    return {
        "event_id": event.get("event_id"),
        "received_at_utc": event.get("received_at_utc"),
        "event_timestamp": event.get("event_timestamp"),
        "device_id": event.get("device_id"),
        "model_version": event.get("model_version"),
        "prediction": event.get("prediction"),
        "score": event.get("score"),
        "threshold": event.get("threshold"),
        "priority": event.get("priority"),
        "window_size": event.get("window_size"),
        "sampling_rate": event.get("sampling_rate"),
        "mean": features.get("mean"),
        "std": features.get("std"),
        "max": features.get("max"),
        "min": features.get("min"),
        "free_heap_kb": features.get("free_heap_kb"),
        "rssi": features.get("rssi"),
        "signal_window": event.get("signal_window"),
        "source": extra.get("source"),
        "firmware_version": extra.get("firmware_version"),
        "tinyml_mode": extra.get("tinyml_mode"),
        "feedback_status": event.get("feedback_status"),
        "human_label": event.get("human_label"),
    }


def flatten_status(status: dict) -> dict:
    extra = status.get("extra") or {}

    return {
        "status_id": status.get("status_id"),
        "received_at_utc": status.get("received_at_utc"),
        "device_id": status.get("device_id"),
        "firmware_version": status.get("firmware_version"),
        "model_version": status.get("model_version"),
        "battery_level": status.get("battery_level"),
        "free_memory_kb": status.get("free_memory_kb"),
        "signal_quality": status.get("signal_quality"),
        "wifi_ip": extra.get("wifi_ip"),
        "chip_model": extra.get("chip_model"),
        "cpu_freq_mhz": extra.get("cpu_freq_mhz"),
        "flash_size_mb": extra.get("flash_size_mb"),
        "sdk_version": extra.get("sdk_version"),
    }


def flatten_device(device: dict) -> dict:
    extra = device.get("extra") or {}

    return {
        "device_id": device.get("device_id"),
        "registered_at_utc": device.get("registered_at_utc"),
        "last_seen_at_utc": device.get("last_seen_at_utc"),
        "device_type": device.get("device_type"),
        "location": device.get("location"),
        "firmware_version": device.get("firmware_version"),
        "model_version": device.get("model_version"),
        "status": device.get("status"),
        "battery_level": device.get("battery_level"),
        "free_memory_kb": device.get("free_memory_kb"),
        "signal_quality": device.get("signal_quality"),
        "wifi_ip": extra.get("wifi_ip"),
        "chip_model": extra.get("chip_model"),
        "cpu_freq_mhz": extra.get("cpu_freq_mhz"),
        "flash_size_mb": extra.get("flash_size_mb"),
        "sdk_version": extra.get("sdk_version"),
        "last_status_id": device.get("last_status_id"),
    }


def export_events(events_dir: Path | None):
    if events_dir is None:
        print("Diretorio de eventos nao encontrado.")
        return

    print(f"Diretorio de eventos encontrado: {events_dir}")

    rows = []

    for path in sorted(events_dir.glob("*.json")):
        data = load_json_file(path)
        if data is not None:
            rows.append(flatten_event(data))

    export_csv(rows, EXPORT_DIR / "events.csv")


def export_status_history(status_history_dir: Path | None):
    if status_history_dir is None:
        print("Diretorio de historico de status nao encontrado.")
        return

    print(f"Diretorio de historico de status encontrado: {status_history_dir}")

    rows = []

    for path in sorted(status_history_dir.glob("*.json")):
        data = load_json_file(path)
        if data is not None:
            rows.append(flatten_status(data))

    export_csv(rows, EXPORT_DIR / "device_status_history.csv")


def export_devices(devices_dir: Path | None):
    if devices_dir is None:
        print("Diretorio de dispositivos nao encontrado.")
        return

    print(f"Diretorio de dispositivos encontrado: {devices_dir}")

    rows = []

    for path in sorted(devices_dir.glob("*.json")):
        data = load_json_file(path)
        if data is not None:
            rows.append(flatten_device(data))

    export_csv(rows, EXPORT_DIR / "devices.csv")


def main():
    print("Exportando dados dos eventos do ESP8266")
    print(f"Base do projeto: {BASE_DIR}")
    print(f"Pasta de saida: {EXPORT_DIR}")
    print()

    events_dir = find_events_dir()
    devices_dir = find_devices_dir()
    status_history_dir = find_status_history_dir(devices_dir)

    export_events(events_dir)
    print()

    export_status_history(status_history_dir)
    print()

    export_devices(devices_dir)
    print()

    print("Exportacao finalizada.")


if __name__ == "__main__":
    main()