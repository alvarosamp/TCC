from pathlib import Path
import numpy as np
from obspy import read
from obspy.core.trace import Trace

RAW_DIR = Path(r"C:\Users\vish8\OneDrive\Documentos\SeriesTemporaisSismicas\data\scedc-pds\event_waveforms\2016\2016_001")
OUT_DIR = Path(r"C:\Users\vish8\OneDrive\Documentos\SeriesTemporaisSismicas\data\processed\windows_40hz_60s")

TARGET_SR = 40.0
WIN_SEC = 60
N_SAMPLES = int(TARGET_SR * WIN_SEC)

PREFERRED_CHANNELS = {"BHZ", "BHN", "BHE", "HNZ", "HNN", "HNE"}

def preprocess_trace(tr: Trace) -> np.ndarray | None:
    tr = tr.copy()

    if tr.stats.channel not in PREFERRED_CHANNELS:
        return None

    tr.detrend("linear")
    tr.detrend("demean")

    # Reamostra para 40 Hz
    if float(tr.stats.sampling_rate) != TARGET_SR:
        tr.resample(TARGET_SR)

    data = tr.data.astype(np.float32)

    # Normalização
    std = float(np.std(data))
    if std < 1e-6:
        return None
    data = (data - float(np.mean(data))) / std

    # Janela fixa de 60s
    if len(data) < N_SAMPLES:
        data = np.pad(data, (0, N_SAMPLES - len(data)), mode="constant")
    else:
        data = data[:N_SAMPLES]

    return data

def main(max_files: int = 10):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ms_files = sorted(RAW_DIR.glob("*.ms"))[:max_files]

    X = []
    meta = []

    for f in ms_files:
        st = read(str(f))
        for tr in st:
            arr = preprocess_trace(tr)
            if arr is None:
                continue
            X.append(arr)
            meta.append({
                "source_file": f.name,
                "station": tr.stats.station,
                "channel": tr.stats.channel,
                "starttime": str(tr.stats.starttime),
                "endtime": str(tr.stats.endtime),
            })

    if not X:
        raise RuntimeError("Nenhuma janela foi gerada. Ajuste os canais ou verifique os arquivos.")

    X = np.stack(X).astype(np.float32)

    np.savez_compressed(OUT_DIR / "windows.npz", X=X)
    np.save(OUT_DIR / "meta.npy", np.array(meta, dtype=object), allow_pickle=True)

    print("✅ Salvo em:", OUT_DIR)
    print("Windows shape:", X.shape)  # (N, 2400)

if __name__ == "__main__":
    main(max_files=10)
