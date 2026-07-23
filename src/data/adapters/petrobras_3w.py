"""
Adapter para o dataset público Petrobras 3W.

Dataset: https://github.com/petrobras/3W
Formato: CSVs com timestamp + colunas de sensores + coluna 'class'
         class==0  → normal
         class>0   → falha (tipo varia por arquivo)

Este adapter:
  1. Lê os CSVs de raw/3w/<tipo_falha>/<poco>.csv
  2. Aplica preprocessing edge-aware por canal
  3. Janelamento com sobreposição configurável
  4. Rótulo binário: 0=normal, 1=qualquer falha
  5. Split por poço (sem data leakage entre poços)
  6. Salva processed/3w/dataset.npz no contrato genérico

Estrutura esperada em raw/3w/:
  raw/3w/
    WELL-00001_20140118.csv
    WELL-00002_20140125.csv
    ...

  Ou organizado por tipo de falha:
    raw/3w/1/WELL-00001.csv     (tipo 1 = abrupt_increase_BSW)
    raw/3w/0/WELL-00002.csv     (tipo 0 = normal)

Cada CSV tem colunas:
  timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP,
  P-JUS-CKGL, T-JUS-CKGL, QGL, class

Referência:
  Vargas et al. (2019) "A Realistic and Public Dataset with Rare
  Undesirable Real Events in Oil Wells". Journal of Petroleum Science
  and Engineering. DOI: 10.1016/j.petrol.2019.106223
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.settings import PROCESSED_DIR, RAW_DIR, SEED

warnings.filterwarnings("ignore")
log = logging.getLogger("3w_adapter")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ============================================================
# CONSTANTES
# ============================================================

RAW_3W_DIR = RAW_DIR / "3w"
PROCESSED_3W_DIR = PROCESSED_DIR / "3w"

# Colunas de sensores do dataset 3W (8 variáveis de processo)
SENSOR_COLUMNS = [
    "P-PDG",
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
    "T-JUS-CKGL",
    "QGL",
]
LABEL_COLUMN = "class"
N_CHANNELS = len(SENSOR_COLUMNS)  # 8


# ============================================================
# LEITURA E PREPROCESSING
# ============================================================

def _load_csv(path: Path) -> pd.DataFrame | None:
    """Carrega um CSV do 3W e retorna DataFrame com sensores e label."""
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    except Exception:
        try:
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
        except Exception as e:
            log.warning(f"Falha ao ler {path}: {e}")
            return None

    missing_sensors = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing_sensors:
        log.warning(f"Colunas ausentes em {path}: {missing_sensors}")
        return None

    if LABEL_COLUMN not in df.columns:
        log.warning(f"Coluna '{LABEL_COLUMN}' ausente em {path}")
        return None

    return df


def _preprocess_channel(x: np.ndarray) -> np.ndarray:
    """
    Preprocessing edge-aware por canal:
      interpolate_missing → detrend_linear → (zscore por janela é feito depois)
    """
    # interpolação linear de NaN
    series = pd.Series(x.astype(np.float32))
    series = series.interpolate(method="linear", limit_direction="both")
    series = series.fillna(0.0)
    x = series.values.astype(np.float32)

    # detrend linear
    n = len(x)
    if n > 1:
        t = np.arange(n, dtype=np.float32)
        slope = (np.cov(t, x)[0, 1]) / (np.var(t) + 1e-8)
        intercept = np.mean(x) - slope * np.mean(t)
        x = x - (slope * t + intercept)

    return x


def _zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalização z-score por canal dentro da janela. window.shape=(W, C)"""
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    return (window - mean) / (std + eps)


def _preprocess_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Processa um DataFrame completo.

    Retorna:
      signals: (T, N_CHANNELS) float32
      labels:  (T,)            int32  — 0 ou 1
    """
    signals = np.stack(
        [_preprocess_channel(df[c].values) for c in SENSOR_COLUMNS],
        axis=1,
    ).astype(np.float32)

    raw_labels = df[LABEL_COLUMN].fillna(0).values.astype(np.int32)
    # binário: qualquer classe > 0 é anomalia
    binary_labels = (raw_labels > 0).astype(np.int32)

    return signals, binary_labels


# ============================================================
# JANELAMENTO
# ============================================================

def _window_signals(
    signals: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Janelamento deslizante.

    signals: (T, C)
    labels:  (T,)
    Retorna:
      X: (N_windows, window_size, C)
      y: (N_windows,)  — label majoritário da janela
    """
    T = len(signals)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        window = signals[start:end]  # (window_size, C)
        window_labels = labels[start:end]

        # z-score por canal por janela
        window = _zscore_window(window)

        # label: se qualquer amostra é anômala → janela anômala
        y = int((window_labels > 0).any())

        X_list.append(window)
        y_list.append(y)

    if not X_list:
        return np.empty((0, window_size, N_CHANNELS), dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


# ============================================================
# COLETA DE ARQUIVOS E SPLIT POR POÇO
# ============================================================

def _discover_csv_files(raw_dir: Path) -> list[Path]:
    """Descobre todos os CSVs recursivamente em raw/3w/."""
    files = list(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {raw_dir}. "
            f"Baixe o dataset 3W em https://github.com/petrobras/3W "
            f"e extraia em {raw_dir}/"
        )
    return sorted(files)


def _split_well_ids(
    well_ids: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Split por poço: treino/val/teste sem sobreposição de poços."""
    rng = np.random.default_rng(seed)
    ids = np.array(well_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_ids = ids[:n_test].tolist()
    val_ids = ids[n_test:n_test + n_val].tolist()
    train_ids = ids[n_test + n_val:].tolist()
    return train_ids, val_ids, test_ids


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def generate_3w_dataset(
    window_size: int = 60,
    step_seconds: float = 30.0,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    fault_types: list[int] | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_normal_ratio: float = 0.05,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Gera o dataset 3W no formato genérico do pipeline.

    Args:
        window_size:       janelas em amostras (default 60 @ 1 Hz = 60 s)
        step_seconds:      passo entre janelas em segundos
        raw_dir:           pasta raiz dos CSVs (default raw/3w/)
        output_dir:        onde salvar o .npz (default processed/3w/)
        fault_types:       lista de tipos de falha a incluir (None = todos)
        val_ratio:         fração de poços para validação
        test_ratio:        fração de poços para teste
        min_normal_ratio:  descarta poços com menos que essa fração de janelas normais
        seed:              semente aleatória para reprodutibilidade

    Returns:
        dict com caminhos de saída e estatísticas do dataset
    """
    raw_dir = raw_dir or RAW_3W_DIR
    output_dir = output_dir or PROCESSED_3W_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, int(step_seconds))  # 1 Hz → step em amostras = step em segundos

    log.info(f"Descobrindo CSVs em {raw_dir} ...")
    csv_files = _discover_csv_files(raw_dir)
    log.info(f"  {len(csv_files)} arquivos encontrados")

    # processa cada CSV e agrupa por ID de poço (nome do arquivo sem extensão)
    well_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for csv_path in csv_files:
        df = _load_csv(csv_path)
        if df is None:
            continue

        # filtrar por tipo de falha se especificado
        if fault_types is not None:
            raw_labels = df[LABEL_COLUMN].fillna(0).values.astype(np.int32)
            has_target_fault = any(
                int((raw_labels == ft).any()) for ft in fault_types
            )
            has_normal = int((raw_labels == 0).any())
            if not (has_target_fault or has_normal):
                continue

        signals, labels = _preprocess_dataframe(df)
        if len(signals) < window_size:
            log.debug(f"  {csv_path.name}: muito curto ({len(signals)} amostras), ignorado")
            continue

        well_id = csv_path.stem
        if well_id in well_data:
            # concatenar múltiplos arquivos do mesmo poço
            prev_sig, prev_lab = well_data[well_id]
            well_data[well_id] = (
                np.concatenate([prev_sig, signals], axis=0),
                np.concatenate([prev_lab, labels], axis=0),
            )
        else:
            well_data[well_id] = (signals, labels)

    if not well_data:
        raise ValueError(
            f"Nenhum dado válido encontrado em {raw_dir}. "
            f"Verifique se o dataset 3W está corretamente extraído."
        )

    log.info(f"  {len(well_data)} poços com dados válidos")

    # janelamento por poço
    well_windows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for well_id, (signals, labels) in well_data.items():
        X, y = _window_signals(signals, labels, window_size, step)
        if len(X) == 0:
            continue
        # filtrar poços quase inteiramente anômalos (sem exemplos normais suficientes)
        normal_ratio = float((y == 0).mean())
        if normal_ratio < min_normal_ratio:
            log.debug(f"  {well_id}: ratio normal={normal_ratio:.2f} < {min_normal_ratio}, ignorado")
            continue
        well_windows[well_id] = (X, y)

    if not well_windows:
        raise ValueError("Nenhum poço com dados válidos após janelamento.")

    well_ids = list(well_windows.keys())
    train_ids, val_ids, test_ids = _split_well_ids(well_ids, val_ratio, test_ratio, seed)

    log.info(f"  Split: {len(train_ids)} treino / {len(val_ids)} val / {len(test_ids)} teste (poços)")

    def _concat_split(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not ids:
            return (
                np.empty((0, window_size, N_CHANNELS), dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )
        Xs = [well_windows[wid][0] for wid in ids]
        ys = [well_windows[wid][1] for wid in ids]
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = _concat_split(train_ids)
    X_val, y_val = _concat_split(val_ids)
    X_test, y_test = _concat_split(test_ids)

    output_path = output_dir / "dataset.npz"
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    stats = {
        "output_path": str(output_path),
        "n_wells_total": len(well_ids),
        "n_wells_train": len(train_ids),
        "n_wells_val": len(val_ids),
        "n_wells_test": len(test_ids),
        "window_size": window_size,
        "n_channels": N_CHANNELS,
        "sensor_columns": SENSOR_COLUMNS,
        "splits": {
            "train": {
                "n_windows": int(len(X_train)),
                "n_anomaly": int((y_train == 1).sum()),
                "anomaly_ratio": float((y_train == 1).mean()) if len(y_train) else 0.0,
                "shape": list(X_train.shape),
            },
            "val": {
                "n_windows": int(len(X_val)),
                "n_anomaly": int((y_val == 1).sum()),
                "anomaly_ratio": float((y_val == 1).mean()) if len(y_val) else 0.0,
                "shape": list(X_val.shape),
            },
            "test": {
                "n_windows": int(len(X_test)),
                "n_anomaly": int((y_test == 1).sum()),
                "anomaly_ratio": float((y_test == 1).mean()) if len(y_test) else 0.0,
                "shape": list(X_test.shape),
            },
        },
    }

    log.info(f"Dataset 3W salvo em {output_path}")
    log.info(f"  Treino:  {stats['splits']['train']['n_windows']} janelas  "
             f"(anomaly={stats['splits']['train']['anomaly_ratio']:.1%})")
    log.info(f"  Val:     {stats['splits']['val']['n_windows']} janelas  "
             f"(anomaly={stats['splits']['val']['anomaly_ratio']:.1%})")
    log.info(f"  Teste:   {stats['splits']['test']['n_windows']} janelas  "
             f"(anomaly={stats['splits']['test']['anomaly_ratio']:.1%})")

    return stats
