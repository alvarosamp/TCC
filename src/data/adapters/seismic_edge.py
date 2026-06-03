from __future__ import annotations

import csv
import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import obspy

from src.core.settings import PROCESSED_DIR, RAW_DIR, SEED, ensure_directories


# ============================================================
# OBJETIVO DO ARQUIVO
# ============================================================
#
# Este arquivo e um ADAPTADOR DE DOMINIO.
#
# Ele sabe lidar com dados sismicos MiniSEED, mas a saida dele
# e generica:
#
#   X_train, y_train
#   X_val, y_val
#   X_test, y_test
#
# Assim, todo o resto do pipeline nao precisa saber que o dado
# veio de sismologia.
#
# Importante:
#   Esta versao NAO usa remove_response.
#   Esta versao NAO exige StationXML.
#
# Motivo:
#   O ESP32 tambem nao usa remove_response.
#   Entao o treino fica mais alinhado com a inferencia embarcada.


warnings.filterwarnings("ignore")
logging.getLogger("obspy").setLevel(logging.ERROR)

log = logging.getLogger("seismic_edge_adapter")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# CAMINHOS
# ============================================================

DIR_EVENTS = RAW_DIR / "events"
DIR_CONTINUOUS = RAW_DIR / "continuous"

OUTPUT_DIR = PROCESSED_DIR


# ============================================================
# PARAMETROS DO PROFILE seismic_edge_v1
# ============================================================

SAMPLING_RATE = 40.0

WINDOW_SECONDS = 20.0
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLING_RATE)

OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1.0 - OVERLAP))

BANDPASS = (0.5, 15.0)

N_NORMAL_TARGET = 150_000

RNG = np.random.default_rng(SEED)

stats: dict[str, int] = defaultdict(int)


# ============================================================
# PREPROCESSING EDGE-COMPATIBLE
# ============================================================

def preprocess_trace_edge(tr):
    """
    Preprocessamento compativel com borda.

    Pipeline:
      1. resample para 40 Hz, se necessario
      2. detrend linear
      3. demean
      4. taper 5%
      5. bandpass 0.5-15 Hz

    Nao faz:
      - remove_response
      - uso de StationXML
      - conversao para unidade fisica VEL

    Essa decisao reduz o training-serving skew:
      treino offline e ESP32 passam a usar pipelines equivalentes.
    """

    try:
        tr = tr.copy()

        if abs(float(tr.stats.sampling_rate) - SAMPLING_RATE) > 0.01:
            tr.resample(SAMPLING_RATE)

        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(max_percentage=0.05)

        tr.filter(
            "bandpass",
            freqmin=BANDPASS[0],
            freqmax=BANDPASS[1],
            zerophase=True,
        )

        return tr

    except Exception:
        return None


def zscore_window(window: np.ndarray) -> np.ndarray | None:
    """
    Normaliza uma janela individualmente.

    Isso evita que o modelo dependa de escala absoluta do sensor.
    """

    mean = float(window.mean())
    std = float(window.std())

    if std < 1e-12:
        return None

    return ((window - mean) / std).astype(np.float32)


# ============================================================
# WINDOWING
# ============================================================

def extract_center_window(data: np.ndarray) -> np.ndarray | None:
    """
    Extrai uma janela central.

    Usado para events/.
    A premissa e que o evento esta aproximadamente no centro do trace.
    """

    n = len(data)

    if n < WINDOW_SIZE:
        return None

    middle = n // 2
    start = middle - WINDOW_SIZE // 2
    end = start + WINDOW_SIZE

    if start < 0 or end > n:
        return None

    return data[start:end]


def extract_sliding_windows(data: np.ndarray) -> list[np.ndarray]:
    """
    Extrai janelas com overlap.

    Usado para continuous/.
    """

    windows = []
    i = 0

    while i + WINDOW_SIZE <= len(data):
        windows.append(data[i:i + WINDOW_SIZE])
        i += STEP_SIZE

    return windows


# ============================================================
# PROCESSAMENTO DE EVENTS
# ============================================================

def process_events() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Processa arquivos anomalos.

    Entrada:
      raw/events/**/*.ms

    Saida:
      X_anomaly: array (n, 800)
      meta: lista com evid, estacao e timestamp
    """

    log.info("=" * 80)
    log.info("FASE 1/3 - Processando EVENTS como anomalias")
    log.info("=" * 80)

    files = sorted(DIR_EVENTS.rglob("*.ms"))
    log.info(f"Arquivos encontrados: {len(files)}")

    X: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    t0 = time.time()

    for file_index, path in enumerate(files, 1):
        try:
            stream = obspy.read(str(path), format="MSEED")
        except Exception:
            stats["events_read_error"] += 1
            continue

        seen_stations = set()

        for tr in stream:
            net_sta = f"{tr.stats.network}.{tr.stats.station}"

            # Mantem uma janela por estacao dentro do mesmo arquivo.
            if net_sta in seen_stations:
                continue

            seen_stations.add(net_sta)

            processed = preprocess_trace_edge(tr)

            if processed is None:
                stats["events_preprocess_error"] += 1
                continue

            data = processed.data.astype(np.float32)

            window = extract_center_window(data)

            if window is None:
                stats["events_short_trace"] += 1
                continue

            normalized = zscore_window(window)

            if normalized is None:
                stats["events_constant_window"] += 1
                continue

            X.append(normalized)

            meta.append(
                {
                    "tipo": "anomalo",
                    "evid": path.stem,
                    "net_sta": net_sta,
                    "timestamp": float(processed.stats.starttime.timestamp),
                }
            )

            stats["events_ok"] += 1

        if file_index % 100 == 0 or file_index == len(files):
            elapsed = time.time() - t0
            rate = file_index / elapsed if elapsed > 0 else 0.0
            eta = (len(files) - file_index) / rate if rate > 0 else 0.0

            log.info(
                f"{file_index}/{len(files)} | "
                f"janelas={len(X)} | "
                f"{rate:.1f} arq/s | "
                f"ETA={eta/60:.1f} min"
            )

    X_array = np.array(X, dtype=np.float32)

    log.info(f"TOTAL events/anomalo: {X_array.shape}")

    return X_array, meta


# ============================================================
# PROCESSAMENTO DE CONTINUOUS
# ============================================================

def process_continuous() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Processa arquivos normais.

    Entrada:
      raw/continuous/**/*.ms

    Saida:
      X_normal: array (n, 800)
      meta: lista com identificador, estacao e timestamp
    """

    log.info("=" * 80)
    log.info("FASE 2/3 - Processando CONTINUOUS como normal")
    log.info("=" * 80)

    files = sorted(DIR_CONTINUOUS.rglob("*.ms"))
    log.info(f"Arquivos encontrados: {len(files)}")

    X: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    t0 = time.time()

    for file_index, path in enumerate(files, 1):
        try:
            stream = obspy.read(str(path), format="MSEED")
        except Exception:
            stats["continuous_read_error"] += 1
            continue

        for tr in stream:
            net_sta = f"{tr.stats.network}.{tr.stats.station}"

            processed = preprocess_trace_edge(tr)

            if processed is None:
                stats["continuous_preprocess_error"] += 1
                continue

            data = processed.data.astype(np.float32)
            windows = extract_sliding_windows(data)
            base_timestamp = float(processed.stats.starttime.timestamp)

            for window_index, window in enumerate(windows):
                normalized = zscore_window(window)

                if normalized is None:
                    stats["continuous_constant_window"] += 1
                    continue

                X.append(normalized)

                meta.append(
                    {
                        "tipo": "normal",
                        "evid": f"cont_{path.stem}_{window_index}",
                        "net_sta": net_sta,
                        "timestamp": (
                            base_timestamp
                            + window_index * STEP_SIZE / SAMPLING_RATE
                        ),
                    }
                )

                stats["continuous_ok"] += 1

        elapsed = time.time() - t0

        log.info(
            f"{file_index}/{len(files)} | "
            f"janelas={len(X)} | "
            f"{elapsed/60:.1f} min"
        )

    log.info(f"Total bruto normal: {len(X)}")

    # Amostragem para manter o mesmo alvo do v3.
    if len(X) > N_NORMAL_TARGET:
        selected = RNG.choice(
            len(X),
            size=N_NORMAL_TARGET,
            replace=False,
        )

        X = [X[i] for i in selected]
        meta = [meta[i] for i in selected]

        log.info(f"Apos amostragem normal: {len(X)}")

    X_array = np.array(X, dtype=np.float32)

    return X_array, meta


# ============================================================
# SPLITS
# ============================================================

def split_temporal(meta_all: list[dict[str, Any]]) -> np.ndarray:
    """
    Split temporal 70/15/15.

    Mantido por comparabilidade historica.
    """

    timestamps = np.array([m["timestamp"] for m in meta_all])
    order = np.argsort(timestamps)

    n = len(timestamps)

    position = np.empty(n, dtype=int)
    position[order] = np.arange(n)

    splits = np.empty(n, dtype=object)

    for i in range(n):
        p = position[i] / n

        if p < 0.70:
            splits[i] = "train"
        elif p < 0.85:
            splits[i] = "val"
        else:
            splits[i] = "test"

    return splits


def split_by_event(meta_all: list[dict[str, Any]]) -> np.ndarray:
    """
    Split por evento.

    Regra:
      - eventos anomalos sao separados por evid
      - janelas normais sao separadas aleatoriamente

    Esse e o split principal do TCC.
    """

    rng = np.random.default_rng(SEED)

    splits = np.empty(len(meta_all), dtype=object)

    tipos = np.array([m["tipo"] for m in meta_all])
    evids = np.array([m["evid"] for m in meta_all])

    anomaly_mask = tipos == "anomalo"
    normal_mask = tipos == "normal"

    event_ids = np.array(sorted(set(evids[anomaly_mask])))
    rng.shuffle(event_ids)

    n_events = len(event_ids)
    n_train = int(n_events * 0.70)
    n_val = int(n_events * 0.15)

    train_events = set(event_ids[:n_train])
    val_events = set(event_ids[n_train:n_train + n_val])
    test_events = set(event_ids[n_train + n_val:])

    for i, evid in enumerate(evids):
        if tipos[i] != "anomalo":
            continue

        if evid in train_events:
            splits[i] = "train"
        elif evid in val_events:
            splits[i] = "val"
        elif evid in test_events:
            splits[i] = "test"
        else:
            raise ValueError(f"Evento sem split definido: {evid}")

    normal_indices = np.where(normal_mask)[0]
    rng.shuffle(normal_indices)

    n_normal = len(normal_indices)
    n_train = int(n_normal * 0.70)
    n_val = int(n_normal * 0.15)

    splits[normal_indices[:n_train]] = "train"
    splits[normal_indices[n_train:n_train + n_val]] = "val"
    splits[normal_indices[n_train + n_val:]] = "test"

    if np.any(splits == None):  # noqa: E711
        raise ValueError("Existem linhas sem split definido.")

    return splits


# ============================================================
# SALVAMENTO
# ============================================================

def save_split(
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
) -> dict[str, Any]:
    """
    Salva um NPZ no contrato generico.

    Chaves:
      X_train, y_train
      X_val, y_val
      X_test, y_test
    """

    payload = {}

    for part in ["train", "val", "test"]:
        mask = splits == part

        payload[f"X_{part}"] = X[mask]
        payload[f"y_{part}"] = y[mask]

    out_path = OUTPUT_DIR / f"dataset_seismic_edge_v1_split_{split_name}.npz"

    np.savez_compressed(out_path, **payload)

    counts: dict[str, Any] = {}

    for part in ["train", "val", "test"]:
        y_part = payload[f"y_{part}"]

        total = int(len(y_part))
        normal = int((y_part == 0).sum())
        anomaly = int((y_part == 1).sum())

        counts[part] = {
            "total": total,
            "normal": normal,
            "anomaly": anomaly,
            "baseline_auc_pr": anomaly / total if total else 0.0,
        }

        log.info(
            f"{split_name}/{part}: "
            f"total={total} normal={normal} "
            f"anomalo={anomaly} "
            f"baseline_auc_pr={counts[part]['baseline_auc_pr']:.4f}"
        )

    log.info(f"NPZ salvo: {out_path}")

    return counts


def save_inventory(
    meta_all: list[dict[str, Any]],
    y: np.ndarray,
    temporal_splits: np.ndarray,
    event_splits: np.ndarray,
) -> None:
    """
    Salva inventario linha-a-linha.

    Esse arquivo e essencial para auditoria e TCC.
    """

    out_csv = OUTPUT_DIR / "inventario_seismic_edge_v1.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "idx",
                "tipo",
                "evid",
                "net_sta",
                "timestamp",
                "split_temporal",
                "split_evento",
            ]
        )

        for i, meta in enumerate(meta_all):
            writer.writerow(
                [
                    i,
                    "anomalo" if y[i] == 1 else "normal",
                    meta["evid"],
                    meta["net_sta"],
                    meta["timestamp"],
                    temporal_splits[i],
                    event_splits[i],
                ]
            )

    log.info(f"Inventario salvo: {out_csv}")


def save_info(
    X_anomaly: np.ndarray,
    X_normal: np.ndarray,
    X_all: np.ndarray,
    temporal_counts: dict[str, Any],
    event_counts: dict[str, Any],
) -> None:
    """
    Salva JSON com metadados do dataset.

    Esse arquivo ajuda a rastrear exatamente como o dataset foi criado.
    """

    info = {
        "dataset_name": "seismic_edge_v1",
        "description": (
            "Dataset sismico edge-compatible, sem remove_response "
            "e sem dependencia de StationXML."
        ),
        "parameters": {
            "sampling_rate": SAMPLING_RATE,
            "window_seconds": WINDOW_SECONDS,
            "window_size": WINDOW_SIZE,
            "overlap": OVERLAP,
            "step_size": STEP_SIZE,
            "bandpass_hz": BANDPASS,
            "normalization": "zscore_per_window",
            "remove_response": False,
            "stationxml_required": False,
            "n_normal_target": N_NORMAL_TARGET,
            "seed": SEED,
            "pipeline": [
                "resample_40hz",
                "detrend_linear",
                "demean",
                "taper_5pct",
                "bandpass_0p5_15hz_zerophase",
                "zscore_per_window",
            ],
        },
        "shapes": {
            "X_anomaly": list(X_anomaly.shape),
            "X_normal": list(X_normal.shape),
            "X_all": list(X_all.shape),
        },
        "counts": {
            "temporal": temporal_counts,
            "evento": event_counts,
        },
        "processing_stats": dict(stats),
    }

    out_json = OUTPUT_DIR / "dataset_seismic_edge_v1_info.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    log.info(f"Info salvo: {out_json}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Executa o adaptador completo.

    Comando:
      python -m src.data.adapters.seismic_edge
    """

    ensure_directories()

    log.info("=" * 80)
    log.info("ADAPTADOR SEISMIC_EDGE_V1")
    log.info("=" * 80)
    log.info(f"DIR_EVENTS:     {DIR_EVENTS}")
    log.info(f"DIR_CONTINUOUS: {DIR_CONTINUOUS}")
    log.info(f"OUTPUT_DIR:     {OUTPUT_DIR}")
    log.info("=" * 80)

    X_anomaly, meta_anomaly = process_events()
    X_normal, meta_normal = process_continuous()

    y_anomaly = np.ones(len(X_anomaly), dtype=np.int8)
    y_normal = np.zeros(len(X_normal), dtype=np.int8)

    X_all = np.concatenate([X_normal, X_anomaly], axis=0)
    y_all = np.concatenate([y_normal, y_anomaly], axis=0)

    meta_all = meta_normal + meta_anomaly

    log.info("=" * 80)
    log.info("FASE 3/3 - Combinando e salvando")
    log.info("=" * 80)
    log.info(f"X_all: {X_all.shape}")
    log.info(f"y_all: {y_all.shape}")
    log.info(f"normal: {len(X_normal)}")
    log.info(f"anomalo: {len(X_anomaly)}")

    temporal_splits = split_temporal(meta_all)
    event_splits = split_by_event(meta_all)

    temporal_counts = save_split(
        split_name="temporal",
        X=X_all,
        y=y_all,
        splits=temporal_splits,
    )

    event_counts = save_split(
        split_name="evento",
        X=X_all,
        y=y_all,
        splits=event_splits,
    )

    save_inventory(
        meta_all=meta_all,
        y=y_all,
        temporal_splits=temporal_splits,
        event_splits=event_splits,
    )

    save_info(
        X_anomaly=X_anomaly,
        X_normal=X_normal,
        X_all=X_all,
        temporal_counts=temporal_counts,
        event_counts=event_counts,
    )

    log.info("=" * 80)
    log.info("DATASET SEISMIC_EDGE_V1 FINALIZADO")
    log.info("=" * 80)


if __name__ == "__main__":
    main()