"""
Gera um dataset sintetico para testar o pipeline sem dados reais.

O dataset imita o formato do seismic_edge_v1:
  - window_size = 800 amostras
  - classes: 0 (normal), 1 (anomalia)
  - anomalias tem amplitude maior e ruido de spike

Saida: data/processed/dataset_seismic_edge_v1_split_evento.npz

Uso:
  python scripts/generate_synthetic_data.py
  python scripts/generate_synthetic_data.py --samples 2000 200 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Configuracao
# --------------------------------------------------------------------------

WINDOW_SIZE = 800          # deve bater com o profile
SAMPLING_RATE = 40.0       # Hz
SEED = 42

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "dataset_seismic_edge_v1_split_evento.npz"
)


# --------------------------------------------------------------------------
# Geradores de sinal
# --------------------------------------------------------------------------

def _make_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano com componente de baixa frequencia — classe normal."""
    noise = rng.normal(0.0, 0.6, size=(n, WINDOW_SIZE)).astype(np.float32)
    # adiciona deriva lenta para dificultar separacao por energia
    t = np.linspace(0, 2 * np.pi, WINDOW_SIZE)
    for i in range(n):
        drift_amp = rng.uniform(0.0, 0.4)
        drift_freq = rng.uniform(0.5, 2.0)
        noise[i] += (drift_amp * np.sin(drift_freq * t)).astype(np.float32)
    return noise


def _make_anomaly(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sinal com burst de amplitude moderada + spike sutil — classe anomalia.

    Anomalias mais proximas do normal para forcar diferenciacao entre modelos.
    """
    X = rng.normal(0.0, 0.6, size=(n, WINDOW_SIZE)).astype(np.float32)

    for i in range(n):
        # burst no meio da janela com amplitude reduzida
        onset = rng.integers(WINDOW_SIZE // 4, 3 * WINDOW_SIZE // 4)
        duration = rng.integers(30, 120)
        end = min(onset + duration, WINDOW_SIZE)
        amp = rng.uniform(0.8, 2.0)
        t = np.linspace(0, np.pi, end - onset)
        X[i, onset:end] += amp * np.sin(t).astype(np.float32)

        # spike pontual mais sutil
        spike_pos = rng.integers(onset, end)
        X[i, spike_pos] += rng.uniform(1.0, 3.0)

    return X


def zscore_per_window(X: np.ndarray) -> np.ndarray:
    """Normalizacao z-score por janela (mesma que o pipeline offline usa)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------
# Gerador principal
# --------------------------------------------------------------------------

def generate_split(
    n_normal: int,
    n_anomaly: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    X_normal = _make_normal(n_normal, rng)
    X_anomaly = _make_anomaly(n_anomaly, rng)

    X = np.concatenate([X_normal, X_anomaly], axis=0)
    y = np.concatenate(
        [np.zeros(n_normal, dtype=np.int8), np.ones(n_anomaly, dtype=np.int8)]
    )

    # embaralha
    idx = rng.permutation(len(y))
    return zscore_per_window(X[idx]), y[idx]


def generate_dataset(
    n_train: int = 1600,
    n_val: int = 200,
    n_test: int = 200,
    anomaly_ratio_train: float = 0.20,
    anomaly_ratio_eval: float = 0.40,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)

    n_anom_train = int(n_train * anomaly_ratio_train)
    n_norm_train = n_train - n_anom_train

    n_anom_val = int(n_val * anomaly_ratio_eval)
    n_norm_val = n_val - n_anom_val

    n_anom_test = int(n_test * anomaly_ratio_eval)
    n_norm_test = n_test - n_anom_test

    X_train, y_train = generate_split(n_norm_train, n_anom_train, rng)
    X_val, y_val = generate_split(n_norm_val, n_anom_val, rng)
    X_test, y_test = generate_split(n_norm_test, n_anom_test, rng)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_dataset(data: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Dataset salvo em: {path}")
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        n_anom = int((y == 1).sum())
        print(f"  {split:5s}  shape={X.shape}  anomalias={n_anom}/{len(y)} "
              f"({100*n_anom/len(y):.0f}%)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera dataset sintetico para o pipeline.")
    p.add_argument(
        "--samples",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[1600, 200, 200],
        help="Numero de amostras por split (default: 1600 200 200)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Caminho de saida (default: {OUTPUT_PATH})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_train, n_val, n_test = args.samples

    print(f"Gerando dataset sintetico  (seed={SEED}, window_size={WINDOW_SIZE})")
    print(f"  train={n_train}  val={n_val}  test={n_test}")

    data = generate_dataset(n_train=n_train, n_val=n_val, n_test=n_test)
    save_dataset(data, args.output)


if __name__ == "__main__":
    main()
