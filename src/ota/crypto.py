"""
Utilitarios criptograficos simples para o fluxo OTA.

Para manter o projeto reproduzivel sem dependencias extras, a assinatura atual
usa HMAC-SHA256 com segredo compartilhado. Em producao, a evolucao recomendada
e assinatura assimetrica (ex.: Ed25519/ECDSA), onde o dispositivo carrega apenas
a chave publica.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_SECRET = "tcc-dev-ota-signing-key-change-me"


def signing_secret() -> bytes:
    return os.environ.get("TCC_OTA_SIGNING_SECRET", DEFAULT_SECRET).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_payload(payload: dict[str, Any]) -> str:
    return hmac.new(signing_secret(), canonical_json(payload), hashlib.sha256).hexdigest()


def verify_payload(payload: dict[str, Any], signature: str) -> bool:
    expected = sign_payload(payload)
    return hmac.compare_digest(expected, signature)
