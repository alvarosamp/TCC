"""
Publica localmente um pacote OTA validado.

Este script simula uma etapa de publicacao OTA.

Entrada:
  artefacts/ota/packages/<model_version>/

Saida:
  artefacts/ota/releases/<model_version>/
  artefacts/ota/releases/latest.json

A ideia e criar uma estrutura semelhante ao que poderia existir em:
  - servidor HTTP;
  - bucket S3/MinIO;
  - GitHub Release;
  - API de atualizacao.

O dispositivo ou backend poderia consultar latest.json para descobrir
qual e a ultima versao disponivel.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR


OTA_DIR = ARTEFACTS_DIR / "ota"
PACKAGES_DIR = OTA_DIR / "packages"
RELEASES_DIR = OTA_DIR / "releases"
LATEST_RELEASE = RELEASES_DIR / "latest.json"


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


def latest_package_dir() -> Path:
    """Retorna o pacote mais recente pela data de modificacao."""
    if not PACKAGES_DIR.exists():
        raise FileNotFoundError(f"Pasta de pacotes nao encontrada: {PACKAGES_DIR}")

    candidates = [p for p in PACKAGES_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Nenhum pacote encontrado em: {PACKAGES_DIR}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_package_dir(package_arg: str | None) -> Path:
    """
    Resolve pacote por nome ou caminho.

    Se nao receber argumento, usa o pacote mais recente.
    """
    if package_arg:
        path = Path(package_arg)
        if not path.is_absolute():
            path = PACKAGES_DIR / package_arg
        return path

    return latest_package_dir()


def require_validated_package(package_dir: Path) -> dict[str, Any]:
    """
    Exige que validation_report.json exista e esteja aprovado.
    """
    validation_path = package_dir / "validation_report.json"

    if not validation_path.exists():
        raise FileNotFoundError(
            "Pacote ainda nao validado. Rode antes:\n"
            "python -m src.ota.validate_ota_package"
        )

    validation = load_json(validation_path)

    if not validation.get("approved", False):
        raise RuntimeError(
            "Pacote reprovado na validacao. "
            f"Relatorio: {validation_path}"
        )

    return validation


def copy_release(package_dir: Path, release_dir: Path) -> None:
    """
    Copia o pacote validado para a pasta de releases.

    Se a release ja existir, ela e substituida.
    """
    if release_dir.exists():
        shutil.rmtree(release_dir)

    shutil.copytree(package_dir, release_dir)


def build_latest_payload(
    release_dir: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria o latest.json apontando para a release publicada.
    """
    model = validation.get("model", {})
    target = validation.get("target", {})
    artifact = validation.get("artifact", {})

    return {
        "schema_version": "1.0.0",
        "published_at_utc": utc_now_iso(),
        "status": "published_local",
        "release_dir": str(release_dir),
        "model": model,
        "target": target,
        "artifact": {
            "path": str(release_dir / Path(artifact["path"]).name),
            "sha256": artifact.get("expected_sha256"),
        },
        "validation": {
            "approved": validation.get("approved"),
            "validated_at_utc": validation.get("validated_at_utc"),
        },
        "notes": [
            "Publicacao local para simular repositorio OTA.",
            "Ainda nao ha servidor HTTP nem download pelo ESP32.",
            "No OTA real, latest.json seria exposto por um endpoint ou storage.",
        ],
    }


def publish_local_release(package_dir: Path) -> Path:
    """
    Publica o pacote validado em artefacts/ota/releases.
    """
    validation = require_validated_package(package_dir)

    model_version = validation["model"]["version"]
    release_dir = RELEASES_DIR / model_version

    copy_release(package_dir, release_dir)

    latest_payload = build_latest_payload(
        release_dir=release_dir,
        validation=validation,
    )

    save_json(LATEST_RELEASE, latest_payload)

    return release_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publica localmente pacote OTA validado.",
    )
    parser.add_argument(
        "--package",
        type=str,
        default=None,
        help=(
            "Nome ou caminho da pasta do pacote. "
            "Se omitido, usa o pacote mais recente."
        ),
    )
    args = parser.parse_args()

    package_dir = resolve_package_dir(args.package)

    release_dir = publish_local_release(package_dir)

    print("=" * 80)
    print("RELEASE OTA LOCAL PUBLICADA")
    print("=" * 80)
    print(f"Pacote: {package_dir}")
    print(f"Release: {release_dir}")
    print(f"Latest:  {LATEST_RELEASE}")


if __name__ == "__main__":
    main()