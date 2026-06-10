"""
Gera manifesto OTA a partir do modelo promovido para producao.

Este script ainda nao faz upload nem atualiza o ESP32. Ele cria um contrato
claro dizendo qual artefato esta pronto para empacotamento, qual threshold deve
acompanhar o modelo, qual preprocessamento e esperado e como validar a
integridade do arquivo.

Entrada:
  artefacts/registry/production_manifest.json  (metadados, metricas, quality gate)
  artefacts/edge/<model_name>_export_manifest.json  (caminho do .tflite int8)

Saida:
  artefacts/ota/ota_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR, EDGE_DIR


REGISTRY_DIR = ARTEFACTS_DIR / "registry"
OTA_DIR = ARTEFACTS_DIR / "ota"

PRODUCTION_MANIFEST = REGISTRY_DIR / "production_manifest.json"
OTA_MANIFEST = OTA_DIR / "ota_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 do artefato para verificacao de integridade."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_model_version(production: dict[str, Any]) -> str:
    profile = production.get("profile", {})
    profile_name = profile.get("profile_name", "unknown_profile")
    model_name = production.get("model_name", "unknown_model")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{profile_name}_{model_name}_{date}"


def load_export_manifest(model_name: str) -> dict[str, Any]:
    path = EDGE_DIR / f"{model_name}_export_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Export manifest nao encontrado: {path}\n"
            "Rode antes: python -m src.export.export_tflite"
        )
    return load_json(path)


def build_ota_manifest(production: dict[str, Any], export: dict[str, Any]) -> dict[str, Any]:
    int8_info = export["exports"]["int8"]
    artifact_path = Path(int8_info["path"])

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artefato int8 nao encontrado: {artifact_path}\n"
            "Rode antes: python -m src.export.export_tflite"
        )

    profile = production.get("profile", {})
    embedded = profile.get("embedded", {})

    artifact_sha256 = sha256_file(artifact_path)
    model_version = build_model_version(production)

    return {
        "schema_version": "1.0.0",
        "created_at_utc": utc_now_iso(),
        "status": "ready_for_packaging",
        "target": {
            "device": embedded.get("target", "esp32"),
            "runtime": embedded.get("runtime", "tensorflow_lite_micro"),
        },
        "ota": {
            "strategy_initial": embedded.get(
                "ota_strategy_initial",
                "firmware_full_image",
            ),
            "strategy_future": embedded.get(
                "ota_strategy_future",
                "separate_model_update",
            ),
            "rollback_required": True,
            "integrity_check": "sha256",
            "signature_required_future": True,
        },
        "artifact": {
            "type": "tflite_int8",
            "path": str(artifact_path),
            "sha256": artifact_sha256,
            "size_bytes": int8_info["size_bytes"],
            "size_kb": int8_info["size_kb"],
            "quantization": "int8",
            "source_model_path": export.get("source_model_path"),
        },
        "model": {
            "name": production.get("model_name"),
            "family": production.get("family"),
            "version": model_version,
            "threshold": production.get("threshold"),
            "parameter_count": production.get("parameter_count"),
        },
        "profile": {
            "name": profile.get("profile_name"),
            "version": profile.get("profile_version"),
            "domain": profile.get("domain"),
            "sampling_rate": profile.get("sampling_rate"),
            "window_size": profile.get("window_size"),
            "window_seconds": profile.get("window_seconds"),
            "step_seconds": profile.get("step_seconds"),
            "preprocessing": profile.get("preprocessing"),
        },
        "quality": {
            "summary_metrics": production.get("summary_metrics"),
            "quality_gate": production.get("quality_gate"),
        },
    }


def main() -> None:
    production = load_json(PRODUCTION_MANIFEST)
    model_name = production.get("model_name")
    export = load_export_manifest(model_name)

    ota_manifest = build_ota_manifest(production, export)
    save_json(OTA_MANIFEST, ota_manifest)

    int8_info = export["exports"]["int8"]
    print("=" * 80)
    print("MANIFESTO OTA GERADO")
    print("=" * 80)
    print(f"Modelo:       {ota_manifest['model']['name']}")
    print(f"Versao:       {ota_manifest['model']['version']}")
    print(f"Destino:      {ota_manifest['target']['device']}")
    print(f"Artefato:     {int8_info['path']} ({int8_info['size_kb']:.1f} KB)")
    print(f"SHA-256:      {ota_manifest['artifact']['sha256']}")
    print(f"Saida:        {OTA_MANIFEST}")


if __name__ == "__main__":
    main()
