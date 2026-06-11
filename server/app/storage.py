import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from src.core.settings import PROJECT_ROOT


SERVER_DATA_DIR = PROJECT_ROOT/'server'/'data'
EVENTS_DIR = SERVER_DATA_DIR/'events'
FEEDBACK_DIR = SERVER_DATA_DIR/'feedback'
DEVICES_DIR = SERVER_DATA_DIR/'devices'
OTA_REPORTS_DIR = SERVER_DATA_DIR/'ota_reports'

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_server_dirs() -> None:
    for path in [
        SERVER_DATA_DIR,
        EVENTS_DIR,
        FEEDBACK_DIR,
        DEVICES_DIR,
        OTA_REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
        

def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    items = []

    for file_path in sorted(path.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            items.append(json.load(f))

    return items


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"