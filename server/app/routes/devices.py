from fastapi import APIRouter

from server.app.schemas import DeviceRegisterRequest, DeviceStatusRequest
from server.app.storage import (
    DEVICES_DIR,
    generate_id,
    load_json,
    save_json,
    utc_now_iso,
)


router = APIRouter()


@router.post("/register")
def register_device(payload: DeviceRegisterRequest):
    device_path = DEVICES_DIR / f"{payload.device_id}.json"

    device = {
        "device_id": payload.device_id,
        "registered_at_utc": utc_now_iso(),
        "device_type": payload.device_type,
        "location": payload.location,
        "firmware_version": payload.firmware_version,
        "model_version": payload.model_version,
        "status": "registered",
    }

    save_json(device_path, device)

    return {
        "registered": True,
        "device_id": payload.device_id,
    }


@router.post("/status")
def update_device_status(payload: DeviceStatusRequest):
    device_path = DEVICES_DIR / f"{payload.device_id}.json"

    if device_path.exists():
        device = load_json(device_path)
    else:
        device = {
            "device_id": payload.device_id,
            "registered_at_utc": utc_now_iso(),
            "status": "auto_registered",
        }

    device["last_seen_at_utc"] = utc_now_iso()
    device["firmware_version"] = payload.firmware_version
    device["model_version"] = payload.model_version
    device["battery_level"] = payload.battery_level
    device["free_memory_kb"] = payload.free_memory_kb
    device["signal_quality"] = payload.signal_quality
    device["extra"] = payload.extra

    save_json(device_path, device)

    return {
        "updated": True,
        "device_id": payload.device_id,
    }


@router.get("")
def list_devices():
    from server.app.storage import list_json_files

    devices = list_json_files(DEVICES_DIR)

    return {
        "count": len(devices),
        "devices": devices,
    }