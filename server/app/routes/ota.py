#Transforma o latest em endpoint para baixar o arquivo mais recente
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from src.core.settings import ARTEFACTS_DIR
from server.app.schemas import OtaReportRequest
from server.app.storage import OTA_REPORTS_DIR, generate_id, load_json, save_json, utc_now_iso

router = APIRouter()

OTA_RELEASE_DIR = Path(ARTEFACTS_DIR) / "ota" / "releases"
LATEST_RELEASE_PATH = OTA_RELEASE_DIR / "latest.json"

@router.get('/latest')
def get_latest_ota_release():
    if not LATEST_RELEASE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Nenhuma release OTA publicada. "
                "Rode antes: python -m src.ota.publish_local_release"
            ),
        )

    latest = load_json(LATEST_RELEASE_PATH)

    return latest

@router.get('/artifact')
def download_latest_artifact():
    if not LATEST_RELEASE_PATH.exists():
        raise HTTPException(
            status_code = 404,
            detail = f'latest.json nao encontrado em {LATEST_RELEASE_PATH}'
        )
    latest = load_json(LATEST_RELEASE_PATH)
    artifact_path = Path(latest["artifact"]["path"])
    if not artifact_path.exists():
        raise HTTPException(
            status_code = 404,
            detail = f'Artefato nao encontrado {artifact_path}'
        )
    return FileResponse(
        path = artifact_path,
        filename = artifact_path.name,
        media_type = 'application/octet-stream'
    )

@router.post("/report")
def report_ota_status(payload: OtaReportRequest):
    report_id = generate_id("ota_report")

    report = {
        "report_id": report_id,
        "received_at_utc": utc_now_iso(),
        "device_id": payload.device_id,
        "previous_model_version": payload.previous_model_version,
        "new_model_version": payload.new_model_version,
        "status": payload.status,
        "message": payload.message,
    }

    save_json(OTA_REPORTS_DIR / f"{report_id}.json", report)

    return {
        "saved": True,
        "report_id": report_id,
    }