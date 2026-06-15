"""
Assina criptograficamente um pacote OTA local.

Entrada:
  artefacts/ota/packages/<model_version>/ota_manifest.json
  artefacts/ota/packages/<model_version>/package_info.json
  artefacts/ota/packages/<model_version>/artifact.tflite

Saida:
  artefacts/ota/packages/<model_version>/signature.json

A assinatura cobre os metadados principais e o SHA-256 do artefato. Por padrao,
usa HMAC-SHA256 com a variavel TCC_OTA_SIGNING_SECRET. Para TCC/local, ha uma
chave de desenvolvimento. Em producao, troque por segredo externo ou assinatura
assimetrica.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR
from src.ota.crypto import sha256_file, sign_payload

OTA_DIR = ARTEFACTS_DIR / "ota"
PACKAGES_DIR = OTA_DIR / "packages"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def latest_package_dir() -> Path:
    if not PACKAGES_DIR.exists():
        raise FileNotFoundError(f"Pasta de pacotes nao encontrada: {PACKAGES_DIR}")
    candidates = [p for p in PACKAGES_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Nenhum pacote encontrado em: {PACKAGES_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_package_dir(package_arg: str | None) -> Path:
    if package_arg:
        path = Path(package_arg)
        if not path.is_absolute():
            path = PACKAGES_DIR / package_arg
        return path
    return latest_package_dir()


def build_signed_payload(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "ota_manifest.json")
    package_info = load_json(package_dir / "package_info.json")
    artifact_path = Path(package_info["artifact"]["packaged_path"])
    actual_sha256 = sha256_file(artifact_path)

    return {
        "schema_version": "1.0.0",
        "package_dir": str(package_dir),
        "model": manifest.get("model"),
        "target": manifest.get("target"),
        "artifact": {
            "packaged_path": str(artifact_path),
            "sha256": actual_sha256,
            "size_bytes": artifact_path.stat().st_size,
            "type": manifest.get("artifact", {}).get("type"),
        },
        "ota": manifest.get("ota"),
    }


def sign_package(package_dir: Path) -> Path:
    signed_payload = build_signed_payload(package_dir)
    signature = sign_payload(signed_payload)
    signature_report = {
        "schema_version": "1.0.0",
        "signed_at_utc": utc_now_iso(),
        "algorithm": "HMAC-SHA256",
        "key_id": "local-dev-key",
        "payload": signed_payload,
        "signature": signature,
        "security_note": (
            "Implementacao local para rastreabilidade. Em producao, preferir "
            "assinatura assimetrica com chave publica no dispositivo."
        ),
    }
    out = package_dir / "signature.json"
    save_json(out, signature_report)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Assina pacote OTA local.")
    parser.add_argument("--package", type=str, default=None)
    args = parser.parse_args()

    package_dir = resolve_package_dir(args.package)
    signature_path = sign_package(package_dir)

    print("=" * 80)
    print("PACOTE OTA ASSINADO")
    print("=" * 80)
    print(f"Pacote:    {package_dir}")
    print(f"Assinatura: {signature_path}")


if __name__ == "__main__":
    main()
