from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA = Path(r"C:\Users\vish8\OneDrive\Documentos\SeriesTemporaisSismicas\data\processed\windows_40hz_60s\windows.npz")
X = np.load(DATA)["X"]

plt.figure()
plt.plot(X[0])
plt.title(f"Example window | shape={X[0].shape}")
plt.xlabel("Sample (40 Hz)")
plt.ylabel("Normalized amplitude")
plt.show()
