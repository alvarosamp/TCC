from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artefacts_dir() -> Path:
    return Path(os.environ.get("TCC_ARTEFACTS_DIR", project_root() / "artefacts"))


def windows_dir() -> Path:
    return Path(os.environ.get("TCC_WINDOWS_DIR", artefacts_dir() / "data" / "windows"))


def results_dir(*parts: str) -> Path:
    base = Path(os.environ.get("TCC_RESULTS_DIR", artefacts_dir() / "results"))
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path | None:
    raw = os.environ.get("TCC_PROCESSED_DIR")
    return Path(raw) if raw else None

