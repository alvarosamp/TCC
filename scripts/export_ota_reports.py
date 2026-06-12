from pathlib import Path
import json
import csv


BASE_DIR = Path(__file__).resolve().parents[1]
OTA_REPORTS_DIR = BASE_DIR / "server" / "data" / "ota_reports"
EXPORT_DIR = BASE_DIR / "data" / "exported"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []

    if not OTA_REPORTS_DIR.exists():
        print(f"Diretorio nao encontrado: {OTA_REPORTS_DIR}")
        return

    for path in sorted(OTA_REPORTS_DIR.glob("*.json")):
        report = load_json(path)

        rows.append({
            "report_id": report.get("report_id"),
            "received_at_utc": report.get("received_at_utc"),
            "device_id": report.get("device_id"),
            "previous_model_version": report.get("previous_model_version"),
            "new_model_version": report.get("new_model_version"),
            "status": report.get("status"),
            "message": report.get("message"),
        })

    output_path = EXPORT_DIR / "ota_reports.csv"

    if not rows:
        print("Nenhum relatorio OTA encontrado.")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exportado: {output_path}")
    print(f"Total de relatorios OTA: {len(rows)}")


if __name__ == "__main__":
    main()