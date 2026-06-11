from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from src.core.settings import ARTEFACTS_DIR
from server.app.schemas import OTAUpdateResponse
from server.app.storage import OTA_REPORTS_DIR, generate_id, load_json, save_json, utc_now_iso

