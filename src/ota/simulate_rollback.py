"""
Simula politica de rollback OTA.

Este script nao atualiza um ESP32 real.
Ele simula o comportamento esperado quando uma atualizacao falha.

Entrada:
  artefacts/ota/device_state.json
  artefacts/ota/releases/latest.json

Saida:
  artefacts/ota/rollback_report.json

Fluxo:
  1. Salva snapshot do estado atual.
  2. Simula inicio de uma instalacao.
  3. Simula falha.
  4. Restaura o estado anterior.
  5. Gera relatorio de rollback.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR


OTA_DIR = ARTEFACTS_DIR / "ota"

DEVICE_STATE = OTA_DIR / "device_state.json"
LATEST_RELEASE = OTA_DIR / "releases" / "latest.json"
ROLLBACK_REPORT = OTA_DIR / "rollback_report.json"


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


def simulate_failed_install(
    current_state: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    """
    Simula um estado intermediario de instalacao com falha.

    Em firmware real, isso poderia representar:
      - download incompleto;
      - checksum invalido;
      - falha ao gravar particao OTA;
      - reboot sem confirmacao;
      - modelo incompatibilidade com runtime.
    """
    failed_state = deepcopy(current_state)

    failed_state["pending_update"] = {
        "status": "failed",
        "failed_at_utc": utc_now_iso(),
        "attempted_model": latest.get("model"),
        "attempted_artifact": latest.get("artifact"),
        "reason": "simulated_install_failure",
    }

    return failed_state


def rollback_to_previous_state(
    previous_state: dict[str, Any],
    failed_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Restaura o estado anterior e registra que houve rollback.
    """
    restored = deepcopy(previous_state)

    restored["last_rollback"] = {
        "rolled_back_at_utc": utc_now_iso(),
        "from_failed_update": failed_state.get("pending_update"),
        "restored_model": previous_state.get("current_model"),
        "status": "rollback_completed",
    }

    return restored


def build_rollback_report(
    previous_state: dict[str, Any],
    failed_state: dict[str, Any],
    restored_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Gera relatorio do rollback.
    """
    return {
        "schema_version": "1.0.0",
        "created_at_utc": utc_now_iso(),
        "status": "rollback_completed",
        "device_id": restored_state.get("device_id"),
        "previous_model": previous_state.get("current_model"),
        "failed_update": failed_state.get("pending_update"),
        "restored_model": restored_state.get("current_model"),
        "rollback": restored_state.get("last_rollback"),
        "notes": [
            "Rollback simulado em device_state.json.",
            "Nenhum ESP32 real foi atualizado nesta etapa.",
            "Em firmware real, o rollback dependeria de particoes OTA e confirmacao de boot.",
        ],
    }


def main() -> None:
    previous_state = load_json(DEVICE_STATE)
    latest = load_json(LATEST_RELEASE)

    failed_state = simulate_failed_install(
        current_state=previous_state,
        latest=latest,
    )

    restored_state = rollback_to_previous_state(
        previous_state=previous_state,
        failed_state=failed_state,
    )

    rollback_report = build_rollback_report(
        previous_state=previous_state,
        failed_state=failed_state,
        restored_state=restored_state,
    )

    save_json(DEVICE_STATE, restored_state)
    save_json(ROLLBACK_REPORT, rollback_report)

    print("=" * 80)
    print("ROLLBACK OTA SIMULADO")
    print("=" * 80)
    print(f"Device: {restored_state.get('device_id')}")
    print(f"Status: {rollback_report['status']}")
    print(f"Modelo restaurado: {restored_state['current_model']['name']}")
    print(f"Versao restaurada: {restored_state['current_model']['version']}")
    print(f"Relatorio: {ROLLBACK_REPORT}")


if __name__ == "__main__":
    main()