"""
plot_window.py — Visualiza a primeira janela do dataset gerado.

Uso: python plot_window.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Caminho relativo à raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed" / "windows_40hz_60s" / "windows.npz"

X = np.load(DATA)["X"]
print(f"Dataset: {X.shape[0]} janelas, {X.shape[1]} amostras cada ({X.shape[1]/40:.0f}s @ 40Hz)")

plt.figure()
plt.plot(X[0])
plt.title(f"Janela exemplo | shape={X[0].shape}")
plt.xlabel("Amostra (40 Hz)")
plt.ylabel("Amplitude normalizada (z-score)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
