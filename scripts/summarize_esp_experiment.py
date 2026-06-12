from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE_DIR / "data" / "exported"
SUMMARY_DIR = BASE_DIR / "data" / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

events_path = EXPORT_DIR / "events.csv"
status_path = EXPORT_DIR / "device_status_history.csv"
devices_path = EXPORT_DIR / "devices.csv"
ota_path = EXPORT_DIR / "ota_reports.csv"

events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
devices = pd.read_csv(devices_path) if devices_path.exists() else pd.DataFrame()
ota = pd.read_csv(ota_path) if ota_path.exists() else pd.DataFrame()


def safe_value_counts(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "Nao disponivel"

    counts = df[column].value_counts()
    return "\n".join([f"- {idx}: {value}" for idx, value in counts.items()])


def safe_describe(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "Nao disponivel"

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return "Nao disponivel"

    return (
        f"- media: {series.mean():.4f}\n"
        f"- desvio_padrao: {series.std():.4f}\n"
        f"- minimo: {series.min():.4f}\n"
        f"- mediana: {series.median():.4f}\n"
        f"- maximo: {series.max():.4f}"
    )


def get_latest_device_row(device_id: str):
    if devices.empty or "device_id" not in devices.columns:
        return None

    filtered = devices[devices["device_id"] == device_id]

    if filtered.empty:
        return None

    return filtered.iloc[-1]


def main():
    target_device = "esp8266_001"
    latest_device = get_latest_device_row(target_device)

    lines = []

    lines.append("# Resumo do experimento ESP8266 - TinyML OTA Feedback Server")
    lines.append("")
    lines.append("## Visao geral")
    lines.append("")
    lines.append(f"- Eventos exportados: {len(events)}")
    lines.append(f"- Amostras historicas de status: {len(status)}")
    lines.append(f"- Dispositivos registrados: {len(devices)}")
    lines.append(f"- Relatorios OTA exportados: {len(ota)}")
    lines.append("")

    lines.append("## Estado atual do dispositivo ESP8266")
    lines.append("")

    if latest_device is not None:
        for col in [
            "device_id",
            "device_type",
            "location",
            "firmware_version",
            "model_version",
            "status",
            "last_seen_at_utc",
            "battery_level",
            "free_memory_kb",
            "signal_quality",
            "wifi_ip",
            "chip_model",
            "cpu_freq_mhz",
            "flash_size_mb",
            "sdk_version",
            "last_status_id",
        ]:
            if col in latest_device.index:
                lines.append(f"- {col}: {latest_device[col]}")
    else:
        lines.append("Dispositivo ESP8266 nao encontrado em devices.csv.")

    lines.append("")

    lines.append("## Distribuicao das predicoes")
    lines.append("")
    lines.append(safe_value_counts(events, "prediction"))
    lines.append("")

    lines.append("## Distribuicao das prioridades")
    lines.append("")
    lines.append(safe_value_counts(events, "priority"))
    lines.append("")

    lines.append("## Estatisticas dos scores")
    lines.append("")
    lines.append(safe_describe(events, "score"))
    lines.append("")

    lines.append("## Estatisticas de memoria livre")
    lines.append("")
    lines.append(safe_describe(status, "free_memory_kb"))
    lines.append("")

    lines.append("## Estatisticas de RSSI")
    lines.append("")
    lines.append(safe_describe(status, "signal_quality"))
    lines.append("")

    lines.append("## Relatorios OTA")
    lines.append("")

    if not ota.empty:
        ota_esp = ota[ota["device_id"] == target_device] if "device_id" in ota.columns else ota

        lines.append(f"- Relatorios OTA do ESP8266: {len(ota_esp)}")

        if not ota_esp.empty:
            last_ota = ota_esp.iloc[-1]
            lines.append(f"- Ultimo status OTA: {last_ota.get('status', 'Nao disponivel')}")
            lines.append(f"- Versao anterior: {last_ota.get('previous_model_version', 'Nao disponivel')}")
            lines.append(f"- Nova versao: {last_ota.get('new_model_version', 'Nao disponivel')}")
            lines.append(f"- Mensagem: {last_ota.get('message', 'Nao disponivel')}")
    else:
        lines.append("Nenhum relatorio OTA exportado.")

    lines.append("")
    lines.append("## Interpretacao tecnica")
    lines.append("")
    lines.append(
        "O experimento validou o fluxo fisico de comunicacao entre o ESP8266 e o servidor FastAPI. "
        "O dispositivo conseguiu registrar-se, enviar status periodico, transmitir eventos de inferencia simulada, "
        "consultar o manifesto OTA, detectar uma nova versao de modelo compativel com o target esp8266 e registrar "
        "o resultado da atualizacao no endpoint /ota/report. Os dados foram persistidos em arquivos JSON e exportados "
        "para CSV, permitindo analise posterior e geracao de graficos."
    )
    lines.append("")
    lines.append(
        "Nesta etapa, a inferencia TinyML ainda e simulada e o OTA foi validado no nivel de manifesto e versionamento. "
        "O download real e a substituicao do artefato do modelo no dispositivo permanecem como evolucao futura do MVP."
    )

    output_path = SUMMARY_DIR / "esp8266_experiment_summary.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Resumo salvo em: {output_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()