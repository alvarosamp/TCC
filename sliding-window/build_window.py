"""
build_window.py — Geração do dataset de janelas para o autoencoder.

Lê arquivos .ms de event_waveforms, aplica o pipeline completo
(detrend → taper → remoção de resposta → bandpass → resample → z-score)
e salva janelas de tamanho fixo (60s @ 40Hz = 2400 amostras) em .npz.

USO:
    python build_window.py              # processa max_files=50
    python build_window.py --max 100    # processa 100 arquivos

SAÍDA:
    data/processed/windows_40hz_60s/windows.npz   →  chave "X" (N, 2400)
    data/processed/windows_40hz_60s/meta.npy       →  array de dicts com metadados
"""

from pathlib import Path
import sys
import numpy as np
from obspy import read, read_inventory
from obspy.core.trace import Trace

# ════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — ajuste aqui se mudar a estrutura de pastas
# ════════════════════════════════════════════════════════════════════════

# Raiz do projeto (2 níveis acima deste script: sliding-window/ → TCC/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pasta com os arquivos .ms brutos (event waveforms do SCEDC)
RAW_DIR = PROJECT_ROOT / "data" / "scedc-pds" / "event_waveforms" / "2016" / "2016_001"

# Pasta com os StationXML (resposta instrumental) — usada para remoção
STATION_XML_DIR = PROJECT_ROOT / "data" / "scedc-pds" / "FDSNstationXML"

# Pasta de saída
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "windows_40hz_60s"

# ════════════════════════════════════════════════════════════════════════
# PARÂMETROS DO PIPELINE
# ════════════════════════════════════════════════════════════════════════
TARGET_SR = 40.0            # Taxa de amostragem alvo (Hz)
WIN_SEC = 60                # Duração da janela (segundos)
N_SAMPLES = int(TARGET_SR * WIN_SEC)  # 2400 amostras por janela

# Canais aceitos (banda larga + high-gain)
PREFERRED_CHANNELS = {"BHZ", "BHN", "BHE", "HNZ", "HNN", "HNE"}

# Pré-filtro para remoção de resposta (f1, f2, f3, f4)
# f1-f2 → transição inferior (remove microsismos oceânicos < 0.5 Hz)
# f3-f4 → transição superior (deve ficar abaixo de Nyquist = 20 Hz)
PRE_FILT = (0.5, 1.0, 18.0, 20.0)


def find_station_xml(network: str, station: str) -> Path | None:
    """
    Procura o StationXML correspondente à estação na pasta STATION_XML_DIR.

    Tenta vários padrões de nome comuns no SCEDC:
      - CI/CI_PASC.xml
      - CI/CI.PASC.xml
      - AZ/AZ.BZN.xml
    Retorna None se não encontrar.
    """
    net_dir = STATION_XML_DIR / network
    if not net_dir.exists():
        return None

    # Padrões comuns de nomeação no SCEDC
    patterns = [
        f"{network}_{station}.xml",
        f"{network}.{station}.xml",
        f"{station}.xml",
    ]
    for pat in patterns:
        candidate = net_dir / pat
        if candidate.exists():
            return candidate

    # Fallback: busca glob
    matches = list(net_dir.glob(f"*{station}*.xml"))
    return matches[0] if matches else None


def preprocess_trace(tr: Trace, inv=None) -> np.ndarray | None:
    """
    Pipeline de pré-processamento completo para um único trace.

    Etapas:
      1. Filtrar canal (só aceita PREFERRED_CHANNELS)
      2. Detrend linear + demean
      3. Taper cosine 5%
      4. Remoção de resposta instrumental (se inv disponível)
      5. Filtro passa-banda 0.5–15 Hz
      6. Reamostragem para 40 Hz
      7. Normalização z-score
      8. Pad/crop para exatamente N_SAMPLES (2400)

    Retorna None se o trace deveria ser descartado (canal errado, sinal
    constante, etc.).
    """
    tr = tr.copy()  # nunca modifica o original

    # ── 1. Filtro de canal ──────────────────────────────────────────────
    if tr.stats.channel not in PREFERRED_CHANNELS:
        return None

    # ── 2. Remoção de tendência e média ─────────────────────────────────
    tr.detrend("linear")
    tr.detrend("demean")

    # ── 3. Taper cosine 5% ─────────────────────────────────────────────
    # Suaviza as bordas do sinal para evitar artefatos na FFT (Gibbs).
    tr.taper(max_percentage=0.05, type="cosine")

    # ── 4. Remoção de resposta instrumental (counts → m/s) ─────────────
    # Sem esta etapa, cada estação tem uma "escala" diferente
    # (depende do sensor + digitalizador), e o modelo aprende
    # "assinaturas de estação" em vez de "forma de onda sísmica".
    if inv is not None:
        try:
            tr.remove_response(
                inventory=inv,
                output="VEL",
                pre_filt=PRE_FILT,
                water_level=60,
            )
        except Exception:
            # Se a resposta não cobrir este canal/época, pula o trace.
            return None

    # ── 5. Filtro passa-banda ───────────────────────────────────────────
    # Mantém 0.5–15 Hz (faixa de eventos regionais).
    tr.filter("bandpass", freqmin=0.5, freqmax=15.0)

    # ── 6. Reamostragem para 40 Hz ─────────────────────────────────────
    if float(tr.stats.sampling_rate) != TARGET_SR:
        tr.resample(TARGET_SR)

    data = tr.data.astype(np.float32)

    # ── 7. Normalização z-score ─────────────────────────────────────────
    std = float(np.std(data))
    if std < 1e-6:
        # Sinal constante (gap, saturação, falha) → descarta
        return None
    data = (data - float(np.mean(data))) / std

    # ── 8. Pad ou crop para janela fixa de 60s (2400 amostras) ──────────
    if len(data) < N_SAMPLES:
        data = np.pad(data, (0, N_SAMPLES - len(data)), mode="constant")
    else:
        data = data[:N_SAMPLES]

    return data


def main(max_files: int = 50):
    """Processa os arquivos .ms e salva windows.npz + meta.npy."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ms_files = sorted(RAW_DIR.glob("*.ms"))[:max_files]
    if not ms_files:
        raise FileNotFoundError(f"Nenhum arquivo .ms encontrado em {RAW_DIR}")

    print(f"Processando {len(ms_files)} arquivos de {RAW_DIR}")

    X = []
    meta = []
    skipped = 0

    for f in ms_files:
        st = read(str(f))

        # Tentar encontrar o StationXML para remoção de resposta
        # (usa a rede/estação do primeiro trace como referência)
        inv = None
        if len(st) > 0:
            net = st[0].stats.network
            sta = st[0].stats.station
            xml_path = find_station_xml(net, sta)
            if xml_path is not None:
                try:
                    inv = read_inventory(str(xml_path))
                except Exception:
                    inv = None

        for tr in st:
            arr = preprocess_trace(tr, inv=inv)
            if arr is None:
                skipped += 1
                continue
            X.append(arr)
            meta.append(
                {
                    "source_file": f.name,
                    "station": tr.stats.station,
                    "channel": tr.stats.channel,
                    "network": tr.stats.network,
                    "starttime": str(tr.stats.starttime),
                    "endtime": str(tr.stats.endtime),
                }
            )

    if not X:
        raise RuntimeError(
            "Nenhuma janela foi gerada. Verifique os canais e os arquivos."
        )

    X = np.stack(X).astype(np.float32)

    np.savez_compressed(OUT_DIR / "windows.npz", X=X)
    np.save(OUT_DIR / "meta.npy", np.array(meta, dtype=object), allow_pickle=True)

    print(f"✅ Salvo em: {OUT_DIR}")
    print(f"   Windows shape: {X.shape}")  # esperado: (N, 2400)
    print(f"   Traces descartados: {skipped}")


if __name__ == "__main__":
    n = 50
    if len(sys.argv) > 1 and sys.argv[1] == "--max":
        n = int(sys.argv[2])
    main(max_files=n)
