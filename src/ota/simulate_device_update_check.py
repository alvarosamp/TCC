"""
Simula um dispositivo verificando se existe atualizacao OTA disponivel.

Este script ainda nao atualiza o ESP32 real.
Ele simula a decisao que o firmware ou um agente OTA precisaria tomar.

Entrada:
  artefacts/ota/device_state.json
  artefacts/ota/releases/latest.json

Saida:
  artefacts/ota/device_update_check_report.json

Fluxo:
  1. Le estado atual do dispositivo.
  2. Le latest.json publicado localmente.
  3. Compara versao atual com versao disponivel.
  4. Verifica compatibilidade de target/runtime.
  5. Decide se uma atualizacao deve ser aplicada.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR


OTA_DIR = ARTEFACTS_DIR / "ota"

DEVICE_STATE = OTA_DIR / "device_state.json"
LATEST_RELEASE = OTA_DIR / "releases" / "latest.json"
UPDATE_CHECK_REPORT = OTA_DIR / "device_update_check_report.json"


def load_json(path: Path) -> dict[str, Any]:
    """Carrega JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Salva JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_default_device_state() -> dict[str, Any]:
    """
    Cria um estado inicial ficticio do dispositivo.

    Em um ESP32 real, essas informacoes poderiam vir de:
      - NVS;
      - firmware metadata;
      - particao OTA;
      - arquivo local;
      - endpoint de device registry.
    """
    return {
        "device_id": "esp32-dev-001",
        "target": {
            "device": "esp32",
            "runtime": "tensorflow_lite_micro",
        },
        "current_model": {
            "name": None,
            "version": None,
            "sha256": None,
            "threshold": None,
        },
        "last_update": None,
    }


def load_or_create_device_state() -> dict[str, Any]:
    """
    Le device_state.json.
    Se ele nao existir, cria um estado inicial.
    """
    if DEVICE_STATE.exists():
        return load_json(DEVICE_STATE)

    state = create_default_device_state()
    save_json(DEVICE_STATE, state)
    return state


def is_compatible(device_state: dict[str, Any], latest: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Verifica se a release disponivel e compativel com o dispositivo.
    """
    reasons: list[str] = []

    device_target = device_state.get("target", {})
    latest_target = latest.get("target", {})

    if device_target.get("device") != latest_target.get("device"):
        reasons.append(
            "device incompativel: "
            f"atual={device_target.get('device')} "
            f"release={latest_target.get('device')}"
        )

    if device_target.get("runtime") != latest_target.get("runtime"):
        reasons.append(
            "runtime incompativel: "
            f"atual={device_target.get('runtime')} "
            f"release={latest_target.get('runtime')}"
        )

    return len(reasons) == 0, reasons


def should_update(device_state: dict[str, Any], latest: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Decide se o dispositivo precisa atualizar.
    """
    reasons: list[str] = []

    compatible, compatibility_reasons = is_compatible(device_state, latest)
    if not compatible:
        return False, compatibility_reasons

    current = device_state.get("current_model", {})
    latest_model = latest.get("model", {})
    latest_artifact = latest.get("artifact", {})

    current_version = current.get("version")
    latest_version = latest_model.get("version")

    current_sha = current.get("sha256")
    latest_sha = latest_artifact.get("sha256")

    if current_version != latest_version:
        reasons.append(
            f"versao diferente: atual={current_version} disponivel={latest_version}"
        )

    if current_sha != latest_sha:
        reasons.append("sha256 diferente entre modelo atual e release disponivel")

    return len(reasons) > 0, reasons


def build_update_check_report(
    device_state: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    """
    Monta relatorio da decisao de update.
    """
    update_available, reasons = should_update(device_state, latest)

    return {
        "schema_version": "1.0.0",
        "checked_at_utc": utc_now_iso(),
        "device_id": device_state.get("device_id"),
        "update_available": update_available,
        "decision_reasons": reasons,
        "current_model": device_state.get("current_model"),
        "available_model": latest.get("model"),
        "available_artifact": latest.get("artifact"),
        "target": {
            "device": device_state.get("target"),
            "release": latest.get("target"),
        },
        "next_action": (
            "download_and_validate_package"
            if update_available
            else "keep_current_version"
        ),
    }


def main() -> None:
    device_state = load_or_create_device_state()
    latest = load_json(LATEST_RELEASE)

    report = build_update_check_report(
        device_state=device_state,
        latest=latest,
    )

    save_json(UPDATE_CHECK_REPORT, report)

    print("=" * 80)
    print("SIMULACAO DE CHECK OTA DO DISPOSITIVO")
    print("=" * 80)
    print(f"Device: {report['device_id']}")
    print(f"Update disponivel: {report['update_available']}")
    print(f"Proxima acao: {report['next_action']}")
    print(f"Relatorio: {UPDATE_CHECK_REPORT}")

    if report["decision_reasons"]:
        print("Motivos:")
        for reason in report["decision_reasons"]:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()