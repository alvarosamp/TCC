from pathlib import Path
from obspy import read

root = Path(r"C:\Users\vish8\OneDrive\Documentos\SeriesTemporaisSismicas\data\scedc-pds\event_waveforms\2016\2016_001")

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
