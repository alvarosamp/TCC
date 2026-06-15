"""
create_cru_dataset.py — Gera versão "crua" do dataset para testar o pipeline
de pré-processamento na borda (ESP32).

Motivação
---------
O dataset_v4_split_evento.npz contém janelas já normalizadas (z-score).
Para testar o pipeline completo no ESP32 (detrend + taper + bandpass + zscore),
precisamos de dados que ainda não foram normalizados.

Como os arquivos MiniSEED originais não estão disponíveis localmente, este
script gera uma versão "crua simulada":

    raw[i] = norm[i] * amplitude + offset + drift[i]

onde:
  amplitude — escala de velocidade sísmica após remoção de resposta
              (log-uniforme entre 1e-7 e 1e-4 m/s, por janela)
  offset    — offset DC pequeno (uniforme entre -10% e +10% da amplitude)
  drift[i]  — deriva linear lenta (inclinação ~ 5% da amplitude / 800 amostras)

Após aplicar o pipeline ESP32 (detrend + taper + bandpass + zscore), o sinal
resultante deve aproximar o dado normalizado original.

Saída
-----
  D:/tCC/processed/dataset_v4_split_evento_cru.npz
    X_train_cru, y_train  (mesmas labels)
    X_val_cru,   y_val
    X_test_cru,  y_test

  D:/tCC/processed/dataset_v4_split_evento_cru_params.npz
    amplitude_train, offset_train, slope_train
    amplitude_val,   offset_val,   slope_val
    amplitude_test,  offset_test,  slope_test
    (para auditoria / verificação da reversibilidade)

Uso
---
    python scripts/create_cru_dataset.py

Depois exporte para o header do ESP32:
    python scripts/export_real_dataset.py \\
        --dataset dataset_v4_split_evento_cru.npz \\
        --split test --classe all --limit 20 --raw
"""

import os
from pathlib import Path

import numpy as np


DATA_DIR = Path(os.environ.get("TCC_PROCESSED_DIR", "D:/tCC/processed"))
SEED = 42
WINDOW_SIZE = 800


def simular_cru(X_norm: np.ndarray, rng: np.random.Generator):
    """Transforma janelas normalizadas em "cruas" com amplitude sísmica realista.

    Parâmetros de simulação:
      - Amplitude: log-uniforme [1e-7, 1e-4] m/s  (velocidade após remove_response)
      - Offset DC: uniforme [-0.1, +0.1] * amplitude
      - Deriva linear: inclinação uniforme [-0.05, +0.05] * amplitude / N
        → simula drift lento do sensor ao longo da janela de 20 s

    Returns
    -------
    X_cru : array (N, WINDOW_SIZE) float32
    params : dict com amplitude, offset, slope por janela
    """
    N = len(X_norm)
    t = np.arange(WINDOW_SIZE, dtype=np.float32)

    # Parâmetros por janela
    log_amp = rng.uniform(np.log10(1e-7), np.log10(1e-4), size=N)
    amplitude = (10.0 ** log_amp).astype(np.float32)         # [1e-7, 1e-4]
    offset    = rng.uniform(-0.10, 0.10, size=N).astype(np.float32) * amplitude
    slope     = rng.uniform(-0.05, 0.05, size=N).astype(np.float32) * amplitude

    X_cru = np.empty_like(X_norm)
    for i in range(N):
        drift = slope[i] * t / WINDOW_SIZE          # deriva ao longo da janela
        X_cru[i] = X_norm[i] * amplitude[i] + offset[i] + drift

    return X_cru, {"amplitude": amplitude, "offset": offset, "slope": slope}


def main():
    npz_in  = DATA_DIR / "dataset_v4_split_evento.npz"
    npz_out = DATA_DIR / "dataset_v4_split_evento_cru.npz"
    par_out = DATA_DIR / "dataset_v4_split_evento_cru_params.npz"

    if not npz_in.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {npz_in}")

    print("=" * 70)
    print("GERANDO DATASET CRU (SIMULADO)")
    print("=" * 70)
    print(f"Entrada : {npz_in}")
    print(f"Saida   : {npz_out}")
    print()

    data = np.load(npz_in)
    rng  = np.random.default_rng(SEED)

    payload = {}
    params  = {}

    for parte in ["train", "val", "test"]:
        X_norm = data[f"X_{parte}"]
        y      = data[f"y_{parte}"]

        # Garantir shape (N, 800) — achata canal extra se houver
        if X_norm.ndim == 3 and X_norm.shape[-1] == 1:
            X_norm = X_norm[:, :, 0]

        print(f"  {parte:5s}: {X_norm.shape}  labels: "
              f"normal={(y==0).sum()}  anomalo={(y==1).sum()}")

        X_cru, p = simular_cru(X_norm, rng)

        payload[f"X_{parte}"] = X_cru
        payload[f"y_{parte}"] = y
        params[f"amplitude_{parte}"] = p["amplitude"]
        params[f"offset_{parte}"]    = p["offset"]
        params[f"slope_{parte}"]     = p["slope"]

        # Estatísticas para verificar a simulação
        print(f"         amplitude: [{p['amplitude'].min():.2e}, {p['amplitude'].max():.2e}]  "
              f"mean={p['amplitude'].mean():.2e}")
        print(f"         X_cru:     min={X_cru.min():.4e}  max={X_cru.max():.4e}  "
              f"std={X_cru.std():.4e}")

    print()
    np.savez_compressed(npz_out, **payload)
    print(f"OK: {npz_out}")

    np.savez_compressed(par_out, **params)
    print(f"OK: {par_out}")

    print()
    print("Dica — para exportar ao ESP32:")
    print("  python scripts/export_real_dataset.py \\")
    print("      --dataset dataset_v4_split_evento_cru.npz \\")
    print("      --split test --classe all --limit 20 --raw")


if __name__ == "__main__":
    main()
