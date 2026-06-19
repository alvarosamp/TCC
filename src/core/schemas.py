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


def normalize_window_shape(X: np.ndarray, expected_size: int) -> np.ndarray:
    '''
    Normaliza e valida o formato de X sem destruir informação multicanal
    
    Formatos aceitos:
    - (N, T): Serie temporal univariada na janela
    - (N, T, C) : Serie temporal multivariada na janela
    
    Onde:
    - N = numero de janelas
    - T = tamanho da janela
    - C = numero de canais/variaveis/sensores
    
    Dataset mutlcianal = X.shape(12000, 2048, 4)
    
    Observação:
    Esta funcao nao faz flatten de canais
    Para modelo neurais conv1d/tcn, o formato(N, T, C) é desejado
    Para modelos classicos, a etapa deve converter o sinal para tabular
    '''
    X = np.asarray(X)
    if X.ndim == 2:
        #Univariado (T, N)
        if X.shape[1] != expected_size:
            raise ValueError(
                f"Tamanho de janela invalido. "
                f"Recebido={X.shape[1]}, esperado={expected_size}. "
                f"Shape completo={X.shape}"
            )
        return X.astype(np.float32, copy=False)

    if X.ndim == 3:
        # Multivariado: (N, T, C)
        if X.shape[1] != expected_size:
            raise ValueError(
                f"Tamanho de janela invalido. "
                f"Recebido={X.shape[1]}, esperado={expected_size}. "
                f"Shape completo={X.shape}. "
                "O formato esperado para multivariado e (N, T, C)."
            )

        if X.shape[2] < 1:
            raise ValueError(
                f"Numero de canais invalido em X: {X.shape[2]}. "
                f"Shape completo={X.shape}"
            )

        return X.astype(np.float32, copy=False)

    raise ValueError(
        f"Formato de X nao suportado: {X.shape}. "
        "Use (N, T) para serie temporal univariada ou "
        "(N, T, C) para serie temporal multivariada. "
        "Se seu dado tiver 4D ou mais, compacte sensores/eixos extras "
        "na dimensao de canais antes de salvar o .npz."
    )
    
    
            
            
    
# Alias temporario para compatibilidade com codigos antigos.
# Depois que todos os imports forem ajustados, pode remover.
def flatten_window(X: np.ndarray, expected_size: int) -> np.ndarray:
    """
    Compatibilidade temporaria.

    Antes essa funcao achatava/removia dimensoes.
    Agora ela apenas chama normalize_window_shape para preservar canais.
    """
    return normalize_window_shape(X, expected_size)

def infer_window_size_from_x(X: np.ndarray) -> int:
    """
    Infere o tamanho da janela temporal de X.
    (N, T) -> T
    (N, T, C) -> T
    """
    X = np.asarray(X)
    if X.ndim in (2, 3):
        return int(X.shape[1])
    raise ValueError(f"Nao foi possivel inferir window_size para X com shape {X.shape}")


def infer_n_channels(X:np.ndarray) -> int:
    """
    Infere o numero de canais do dataset
    (N, T) -> 1 canal
    (N, T, C) -> C canais
    
    """
    X = np.asarray(X)
    if X.ndim ==2:
        return 1
    
    if X.ndim ==3:
        return int(X.shape[2])
    
    raise ValueError(f'Nao foi possivel inferir canais para X com shape {X.shape}')


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

    x = normalize_window_shape(data[x_key], profile.window_size)
    y = validate_labels(data[y_key], profile, split_name)

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
      - comparar diferentes versoes do dataset
      - identificar se o dado e univariado ou multivariado
    """
    total = int(len(y))
    normal = int((y == profile.normal_label).sum())
    anomaly = int((y == profile.anomaly_label).sum())

    x_ndim = int(x.ndim)
    window_size = infer_window_size_from_x(x)
    n_channels = infer_n_channels(x)

    if x_ndim == 2:
        input_type = "univariate_timeseries"
    elif x_ndim == 3:
        input_type = "multivariate_timeseries"
    else:
        input_type = "unsupported"

    return {
        "total": total,
        "normal": normal,
        "anomaly": anomaly,
        "baseline_auc_pr": anomaly / total if total else 0.0,
        "x_shape": list(x.shape),
        "x_ndim": x_ndim,
        "input_type": input_type,
        "window_size": window_size,
        "n_channels": n_channels,
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
