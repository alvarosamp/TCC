"""
quick_read.py — Leitura rápida de um arquivo .ms para inspeção.

Uso: python quick_read.py
Mostra metadados do primeiro trace do primeiro arquivo encontrado.
"""
from pathlib import Path
from obspy import read

# Raiz do projeto (mesma pasta deste script)
PROJECT_ROOT = Path(__file__).resolve().parent
root = PROJECT_ROOT / "data" / "scedc-pds" / "event_waveforms" / "2016" / "2016_001"

files = sorted([p for p in root.glob("*.ms") if p.is_file()])
print("Files:", len(files))
print("Example:", files[0])

st = read(str(files[0]))
print(st)
tr = st[0]
print("Station:", tr.stats.station, "Channel:", tr.stats.channel)
print("Sampling rate:", tr.stats.sampling_rate)
print("Start:", tr.stats.starttime, "End:", tr.stats.endtime)
print("NPTS:", tr.stats.npts)
