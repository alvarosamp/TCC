"""
Cria um pacote OTA local a partir do ota_manifest.json.

Este script ainda nao faz upload e ainda nao atualiza o ESP32.
Ele apenas organiza os artefatos em uma pasta versionada.

Entrada:
  artefacts/ota/ota_manifest.json

Saida:
  artefacts/ota/packages/<model_version>/
    - ota_manifest.json
    - artifact.<extensao>
    - package_info.json

Por enquanto, o artefato pode ser .keras, pois estamos validando o fluxo.
Na etapa final de TinyML, o ideal sera empacotar .tflite ou firmware .bin.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from src.core.settings import ARTEFACTS_DIR

OTA_DIR = ARTEFACTS_DIR / "ota"
OTA_MANIFEST = OTA_DIR / "ota_manifest.json"
PACKAGES_DIR = OTA_DIR / "packages"

def load_json(path: Path) -> dict[str, Any]:
    """Carrega um arquivo JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Salva um dicionario em JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def copy_artifact(source: Path, package_dir: Path) -> Path:
    """
    Copia o artefato para dentro do pacote.

    O nome padronizado ajuda o firmware/backend a saber qual arquivo consumir.
    """
    if not source.exists():
        raise FileNotFoundError(f"Artefato nao encontrado: {source}")

    suffix = source.suffix.lower()
    if not suffix:
        suffix = ".bin"

    target = package_dir / f"artifact{suffix}"
    shutil.copy2(source, target)

    return target

def build_package_info(
    ota_manifest: dict[str, Any],
    package_dir: Path,
    packaged_artifact: Path,
) -> dict[str, Any]:
    """
    Cria metadados do pacote OTA.

    Este arquivo descreve a pasta do pacote, o artefato copiado
    e o estado do empacotamento.
    """
    artifact = ota_manifest["artifact"]
    model = ota_manifest["model"]
    target = ota_manifest["target"]

    return {
        "schema_version": "1.0.0",
        "created_at_utc": utc_now_iso(),
        "package_status": "created",
        "package_dir": str(package_dir),
        "target": target,
        "model": model,
        "artifact": {
            "original_path": artifact["path"],
            "packaged_path": str(packaged_artifact),
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
            "type": artifact["type"],
        },
        "notes": [
            "Pacote local criado para rastreabilidade.",
            "Ainda nao foi feito upload para servidor OTA.",
            "Para TinyML final, preferir empacotar .tflite ou firmware .bin.",
        ],
    }

def build_ota_package(ota_manifest: dict[str, Any]) -> Path:
    """
    Cria a pasta versionada do pacote OTA.
    """
    model_version = ota_manifest["model"]["version"]
    artifact_path = Path(ota_manifest["artifact"]["path"])

    package_dir = PACKAGES_DIR / model_version
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest_target = package_dir / "ota_manifest.json"
    save_json(manifest_target, ota_manifest)

    packaged_artifact = copy_artifact(
        source=artifact_path,
        package_dir=package_dir,
    )

    package_info = build_package_info(
        ota_manifest=ota_manifest,
        package_dir=package_dir,
        packaged_artifact=packaged_artifact,
    )

    save_json(package_dir / "package_info.json", package_info)

    return package_dir


def main() -> None:
    ota_manifest = load_json(OTA_MANIFEST)

    package_dir = build_ota_package(ota_manifest)

    print("=" * 80)
    print("PACOTE OTA CRIADO")
    print("=" * 80)
    print(f"Modelo:  {ota_manifest['model']['name']}")
    print(f"Versao:  {ota_manifest['model']['version']}")
    print(f"Pacote:  {package_dir}")
    print()
    print("Arquivos esperados:")
    print(f"  - {package_dir / 'ota_manifest.json'}")
    print(f"  - {package_dir / 'package_info.json'}")
    print(f"  - {package_dir / ('artifact' + Path(ota_manifest['artifact']['path']).suffix)}")


if __name__ == "__main__":
    main()