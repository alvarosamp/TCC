from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from src.core.profile import PipelineProfile

REQUIRED_NPZ_KEYS = ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val']

def validate_npz_keys(npz_path: str | Path) -> list[str]:
    """Valida se temos as chaves necessarias"""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Arquivo npz não encontrado: {npz_path}")
    
    data = np.load(npz_path)
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
    if missing:
        raise ValueError(f"Chaves faltando no arquivo npz: {missing}")
    return list(data.files)

def flatten_window(X: np.ndarray, expected_size: int) -> np.ndarray:
    """
    Retorna X em formato 2D (n_janelas, window_size * n_channels) para modelos clássicos.

    Univariado  (n, w)       -> (n, w)
    Multivariado (n, w, c)   -> (n, w*c)   (canais concatenados)
    """
    x = np.asarray(X, dtype=np.float32)

    if x.ndim == 2:
        if x.shape[1] != expected_size:
            raise ValueError(
                f"Tamanho de janela invalido. Recebido={x.shape[1]}, esperado={expected_size}"
            )
        return x

    if x.ndim == 3:
        if x.shape[1] != expected_size:
            raise ValueError(
                f"Tamanho de janela invalido. Recebido={x.shape[1]}, esperado={expected_size}"
            )
        n = x.shape[0]
        return x.reshape(n, -1)

    raise ValueError(f"Formato de X nao suportado: {x.shape}")


def prepare_for_neural(X: np.ndarray, expected_size: int, n_channels: int = 1) -> np.ndarray:
    """
    Retorna X em formato 3D (n_janelas, window_size, n_channels) para redes neurais.

    Univariado  (n, w)       -> (n, w, 1)
    Multivariado (n, w, c)   -> (n, w, c)
    """
    x = np.asarray(X, dtype=np.float32)

    if x.ndim == 2:
        if x.shape[1] != expected_size:
            raise ValueError(
                f"Tamanho de janela invalido. Recebido={x.shape[1]}, esperado={expected_size}"
            )
        return x[:, :, np.newaxis]

    if x.ndim == 3:
        if x.shape[1] != expected_size:
            raise ValueError(
                f"Tamanho de janela invalido. Recebido={x.shape[1]}, esperado={expected_size}"
            )
        if x.shape[2] != n_channels:
            raise ValueError(
                f"Numero de canais invalido. Recebido={x.shape[2]}, esperado={n_channels}"
            )
        return x

    raise ValueError(f"Formato de X nao suportado: {x.shape}")

def validate_labels(
    y: np.ndarray,
    profile: PipelineProfile,
    split_name: str,
) -> np.ndarray:
    """
    Valida se y contem apenas os labels declarados no profile.

    Para seu caso:
      normal  = 0
      anomalo = 1
    """

    y = np.asarray(y).astype(np.int32, copy=False)

    valid_labels = {
        profile.normal_label,
        profile.anomaly_label,
    }

    found_labels = set(np.unique(y).tolist())
    invalid = found_labels - valid_labels

    if invalid:
        raise ValueError(
            f"Labels invalidos no split {split_name}: {invalid}. "
            f"Esperado apenas: {valid_labels}"
        )

    return y


def load_validated_split(
    npz_path: str | Path,
    split_name: str,
    profile: PipelineProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega e valida um split especifico.

    Exemplo:
      split_name='train'
      carrega X_train e y_train
    """

    data = np.load(npz_path)

    x_key = f"X_{split_name}"
    y_key = f"y_{split_name}"

    if x_key not in data.files or y_key not in data.files:
        raise KeyError(f"Chaves ausentes: {x_key}/{y_key}")

    x = flatten_window(data[x_key], profile.window_size)
    y = validate_labels(data[y_key], profile, split_name)

    profile.validate_window_size(x.shape)

    if len(x) != len(y):
        raise ValueError(
            f"Tamanho inconsistente em {split_name}: "
            f"X={len(x)}, y={len(y)}"
        )

    return x, y


def summarize_split(
    x: np.ndarray,
    y: np.ndarray,
    profile: PipelineProfile,
) -> dict[str, Any]:
    """
    Cria um resumo estatistico simples do split.

    Esse resumo e importante para:
      - relatorio
      - MLflow
      - detectar dataset errado
      - comparar v3 vs edge_v1
    """

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


def validate_full_dataset(
    npz_path: str | Path,
    profile: PipelineProfile,
) -> dict[str, Any]:
    """
    Valida o dataset completo.

    Essa funcao deve ser rodada antes de qualquer treino.
    """

    npz_path = Path(npz_path)
    validate_npz_keys(npz_path)

    report: dict[str, Any] = {
        "dataset": str(npz_path),
        "profile": profile.to_dict(),
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        x, y = load_validated_split(
            npz_path=npz_path,
            split_name=split_name,
            profile=profile,
        )

        report["splits"][split_name] = summarize_split(
            x=x,
            y=y,
            profile=profile,
        )

    return report
