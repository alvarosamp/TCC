"""
Valida um pacote OTA local.

Este script verifica se o pacote OTA esta completo e integro.

Entrada:
  artefacts/ota/packages/<model_version>/

Saida:
  artefacts/ota/packages/<model_version>/validation_report.json

Verificacoes:
  - package_info.json existe;
  - ota_manifest.json existe;
  - artefato existe;
  - SHA-256 do artefato bate com o manifesto;
  - target esperado e ESP32;
  - runtime esperado e TensorFlow Lite Micro;
  - status do manifesto e compativel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR, OTA_CONFIG
from src.ota.crypto import verify_payload


OTA_DIR = ARTEFACTS_DIR / "ota"
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


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de um arquivo."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_package_dir() -> Path:
    """
    Retorna o pacote mais recente pela data de modificacao.

    Isso facilita rodar:
      python -m src.ota.validate_ota_package

    sem precisar passar a versao manualmente.
    """
    if not PACKAGES_DIR.exists():
        raise FileNotFoundError(f"Pasta de pacotes nao encontrada: {PACKAGES_DIR}")

    candidates = [p for p in PACKAGES_DIR.iterdir() if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(f"Nenhum pacote encontrado em: {PACKAGES_DIR}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "value": value,
            "expected": expected,
        }
    )


def validate_package(package_dir: Path) -> dict[str, Any]:
    """
    Valida um pacote OTA.
    """
    checks: list[dict[str, Any]] = []

    package_info_path = package_dir / "package_info.json"
    manifest_path = package_dir / "ota_manifest.json"

    add_check(
        checks,
        name="package_dir_exists",
        passed=package_dir.exists() and package_dir.is_dir(),
        value=str(package_dir),
        expected="existing directory",
    )

    add_check(
        checks,
        name="package_info_exists",
        passed=package_info_path.exists(),
        value=str(package_info_path),
        expected="existing file",
    )

    add_check(
        checks,
        name="ota_manifest_exists",
        passed=manifest_path.exists(),
        value=str(manifest_path),
        expected="existing file",
    )

    if not package_info_path.exists() or not manifest_path.exists():
        return {
            "validated_at_utc": utc_now_iso(),
            "package_dir": str(package_dir),
            "approved": False,
            "checks": checks,
            "reasons": [
                "Arquivos obrigatorios ausentes. "
                "Nao foi possivel continuar a validacao."
            ],
        }

    package_info = load_json(package_info_path)
    manifest = load_json(manifest_path)

    artifact_path = Path(package_info["artifact"]["packaged_path"])
    expected_sha256 = manifest["artifact"]["sha256"]

    add_check(
        checks,
        name="artifact_exists",
        passed=artifact_path.exists(),
        value=str(artifact_path),
        expected="existing file",
    )

    if artifact_path.exists():
        actual_sha256 = sha256_file(artifact_path)
    else:
        actual_sha256 = None

    add_check(
        checks,
        name="artifact_sha256_matches",
        passed=actual_sha256 == expected_sha256,
        value=actual_sha256,
        expected=expected_sha256,
    )

    add_check(
        checks,
        name="target_device",
        passed=manifest["target"]["device"] == "esp32",
        value=manifest["target"]["device"],
        expected="esp32",
    )

    add_check(
        checks,
        name="target_runtime",
        passed=manifest["target"]["runtime"] == "tensorflow_lite_micro",
        value=manifest["target"]["runtime"],
        expected="tensorflow_lite_micro",
    )

    add_check(
        checks,
        name="manifest_status",
        passed=manifest["status"] in {"ready_for_packaging", "packaged"},
        value=manifest["status"],
        expected="ready_for_packaging or packaged",
    )

    add_check(
        checks,
        name="model_version_matches_package",
        passed=manifest["model"]["version"] in str(package_dir),
        value=manifest["model"]["version"],
        expected=str(package_dir),
    )

    signature_path = package_dir / "signature.json"
    signature_required = bool(OTA_CONFIG.get("signature_required", True))
    signature_payload = None
    signature_valid = False

    add_check(
        checks,
        name="signature_exists",
        passed=signature_path.exists() or not signature_required,
        value=str(signature_path) if signature_path.exists() else None,
        expected="existing signature.json" if signature_required else "optional",
    )

    if signature_path.exists():
        signature_report = load_json(signature_path)
        signature_payload = signature_report.get("payload")
        signature_valid = verify_payload(
            signature_payload,
            signature_report.get("signature", ""),
        ) if isinstance(signature_payload, dict) else False
        add_check(
            checks,
            name="signature_valid",
            passed=signature_valid,
            value=signature_report.get("algorithm"),
            expected="valid HMAC-SHA256 signature",
        )

    max_tflite_kb = OTA_CONFIG.get("max_tflite_int8_size_kb")
    if max_tflite_kb is not None:
        artifact_size_kb = manifest["artifact"].get("size_kb")
        add_check(
            checks,
            name="max_tflite_int8_size_kb",
            passed=artifact_size_kb is not None and artifact_size_kb <= float(max_tflite_kb),
            value=artifact_size_kb,
            expected=f"<= {max_tflite_kb}",
        )

    approved = all(check["passed"] for check in checks)

    reasons = [
        f"{check['name']} falhou: valor={check['value']} esperado={check['expected']}"
        for check in checks
        if not check["passed"]
    ]

    return {
        "schema_version": "1.0.0",
        "validated_at_utc": utc_now_iso(),
        "package_dir": str(package_dir),
        "approved": approved,
        "model": manifest.get("model"),
        "target": manifest.get("target"),
        "artifact": {
            "path": str(artifact_path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
        },
        "signature": {
            "path": str(package_dir / "signature.json"),
            "required": bool(OTA_CONFIG.get("signature_required", True)),
            "valid": signature_valid,
        },
        "checks": checks,
        "reasons": reasons,
    }


def resolve_package_dir(package_arg: str | None) -> Path:
    """
    Resolve a pasta do pacote.

    Se o usuario passar --package, usa esse caminho.
    Se nao passar, usa o pacote mais recente.
    """
    if package_arg:
        path = Path(package_arg)
        if not path.is_absolute():
            path = PACKAGES_DIR / package_arg
        return path

    return latest_package_dir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida pacote OTA local.",
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

    report = validate_package(package_dir)

    report_path = package_dir / "validation_report.json"
    save_json(report_path, report)

    print("=" * 80)
    print("VALIDACAO DO PACOTE OTA")
    print("=" * 80)
    print(f"Pacote:  {package_dir}")
    print(f"Aprovado: {report['approved']}")
    print(f"Relatorio: {report_path}")

    if not report["approved"]:
        print("Motivos:")
        for reason in report["reasons"]:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()