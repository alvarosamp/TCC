from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE_DIR / "data" / "exported"
FIGURES_DIR = BASE_DIR / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

events_path = EXPORT_DIR / "events.csv"
status_path = EXPORT_DIR / "device_status_history.csv"

events = pd.read_csv(events_path)
status = pd.read_csv(status_path)

events["received_at_utc"] = pd.to_datetime(events["received_at_utc"], errors="coerce")
status["received_at_utc"] = pd.to_datetime(status["received_at_utc"], errors="coerce")

events = events.sort_values("received_at_utc")
status = status.sort_values("received_at_utc")


plt.figure(figsize=(10, 5))
plt.plot(events["received_at_utc"], events["score"], marker="o")
plt.title("Score das inferências ao longo do tempo")
plt.xlabel("Tempo")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "score_over_time.png", dpi=300)
plt.close()


prediction_counts = events["prediction"].value_counts()

plt.figure(figsize=(7, 5))
prediction_counts.plot(kind="bar")
plt.title("Quantidade de eventos por predição")
plt.xlabel("Predição")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "prediction_counts.png", dpi=300)
plt.close()


priority_counts = events["priority"].value_counts()

plt.figure(figsize=(7, 5))
priority_counts.plot(kind="bar")
plt.title("Quantidade de eventos por prioridade")
plt.xlabel("Prioridade")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "priority_counts.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(status["received_at_utc"], status["signal_quality"], marker="o")
plt.title("Qualidade do sinal Wi-Fi ao longo do tempo")
plt.xlabel("Tempo")
plt.ylabel("RSSI (dBm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rssi_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(status["received_at_utc"], status["free_memory_kb"], marker="o")
plt.title("Memória livre do ESP8266 ao longo do tempo")
plt.xlabel("Tempo")
plt.ylabel("Memória livre (KB)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "free_memory_over_time.png", dpi=300)
plt.close()


print(f"Graficos salvos em: {FIGURES_DIR}")
print("Arquivos gerados:")
for path in sorted(FIGURES_DIR.glob("*.png")):
    print("-", path)