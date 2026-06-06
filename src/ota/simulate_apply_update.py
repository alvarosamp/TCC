"""
Simula a aplicacao de uma atualizacao OTA no dispositivo.

Este script ainda nao atualiza um ESP32 real.
Ele apenas atualiza o estado simulado do dispositivo.

Entrada:
  artefacts/ota/device_state.json
  artefacts/ota/device_update_check_report.json
  artefacts/ota/releases/latest.json

Saidas:
  artefacts/ota/device_state.json
  artefacts/ota/install_report.json

Fluxo:
  1. Verifica se existe update disponivel.
  2. Le latest.json.
  3. Simula instalacao do modelo.
  4. Atualiza device_state.json.
  5. Gera install_report.json.
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
INSTALL_REPORT = OTA_DIR / "install_report.json"


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


def assert_update_available(update_report: dict[str, Any]) -> None:
    """
    Garante que o check OTA indicou update disponivel.
    """
    if not update_report.get("update_available", False):
        raise RuntimeError(
            "Nenhuma atualizacao disponivel. "
            "Rode primeiro: python -m src.ota.simulate_device_update_check"
        )

    if update_report.get("next_action") != "download_and_validate_package":
        raise RuntimeError(
            "Proxima acao nao permite instalacao. "
            f"next_action={update_report.get('next_action')}"
        )


def build_updated_device_state(
    current_state: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    """
    Atualiza o estado do dispositivo para refletir a nova versao instalada.
    """
    model = latest["model"]
    artifact = latest["artifact"]

    updated = dict(current_state)
    updated["current_model"] = {
        "name": model.get("name"),
        "version": model.get("version"),
        "sha256": artifact.get("sha256"),
        "threshold": model.get("threshold"),
        "artifact_path": artifact.get("path"),
    }
    updated["last_update"] = {
        "updated_at_utc": utc_now_iso(),
        "source": "local_release_simulation",
        "status": "installed",
    }

    return updated


def build_install_report(
    old_state: dict[str, Any],
    new_state: dict[str, Any],
    latest: dict[str, Any],
    update_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Gera relatorio da instalacao simulada.
    """
    return {
        "schema_version": "1.0.0",
        "installed_at_utc": utc_now_iso(),
        "status": "installed_simulated",
        "device_id": new_state.get("device_id"),
        "previous_model": old_state.get("current_model"),
        "installed_model": new_state.get("current_model"),
        "release": {
            "release_dir": latest.get("release_dir"),
            "target": latest.get("target"),
            "artifact": latest.get("artifact"),
        },
        "decision_reasons": update_report.get("decision_reasons", []),
        "notes": [
            "Instalacao simulada em device_state.json.",
            "Nenhum ESP32 real foi atualizado nesta etapa.",
            "No firmware real, esta etapa envolveria download, escrita em particao OTA e reboot controlado.",
        ],
    }


def main() -> None:
    device_state = load_json(DEVICE_STATE)
    latest = load_json(LATEST_RELEASE)
    update_report = load_json(UPDATE_CHECK_REPORT)

    assert_update_available(update_report)

    updated_state = build_updated_device_state(
        current_state=device_state,
        latest=latest,
    )

    install_report = build_install_report(
        old_state=device_state,
        new_state=updated_state,
        latest=latest,
        update_report=update_report,
    )

    save_json(DEVICE_STATE, updated_state)
    save_json(INSTALL_REPORT, install_report)

    print("=" * 80)
    print("ATUALIZACAO OTA SIMULADA APLICADA")
    print("=" * 80)
    print(f"Device: {updated_state.get('device_id')}")
    print(f"Modelo instalado: {updated_state['current_model']['name']}")
    print(f"Versao instalada: {updated_state['current_model']['version']}")
    print(f"Relatorio: {INSTALL_REPORT}")


if __name__ == "__main__":
    main()