from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .profile import PipelineProfile


def flatten_windows(x: np.ndarray, expected_size: int) -> np.ndarray:
    """Normaliza shapes comuns para (n_janelas, window_size)."""

    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    elif x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Formato de X nao suportado: {x.shape}")
    if x.shape[1] != expected_size:
        raise ValueError(
            f"Janela com tamanho {x.shape[1]}, esperado {expected_size}"
        )
    return x.astype(np.float32, copy=False)


def load_npz_split(npz_path: str | Path, split: str, profile: PipelineProfile):
    data = np.load(npz_path)
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    if x_key not in data.files or y_key not in data.files:
        raise KeyError(f"Chaves ausentes no NPZ: {x_key}/{y_key}")
    x = flatten_windows(data[x_key], profile.window_size)
    y = np.asarray(data[y_key]).astype(np.int32, copy=False)
    profile.validate_window_shape(x.shape)
    return x, y


def summarize_split(x: np.ndarray, y: np.ndarray, profile: PipelineProfile) -> dict[str, Any]:
    total = int(len(y))
    normal = int((y == profile.normal_label).sum())
    anomaly = int((y == profile.anomaly_label).sum())

    return {
        "total": total,
        "normal": normal,
        "anomaly": anomaly,
        "baseline_auc_pr": anomaly / total if total else 0.0,
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "x_mean": float(x.mean()) if total else 0.0,
        "x_std": float(x.std()) if total else 0.0,
        "x_min": float(x.min()) if total else 0.0,
        "x_max": float(x.max()) if total else 0.0,
    }


def inspect_npz_dataset(npz_path: str | Path, profile: PipelineProfile) -> dict[str, Any]:
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"NPZ invalido. Chaves ausentes: {missing}")

    report: dict[str, Any] = {
        "dataset": str(npz_path),
        "profile": profile.to_dict(),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        x, y = load_npz_split(npz_path, split, profile)
        report["splits"][split] = summarize_split(x, y, profile)
    return report
