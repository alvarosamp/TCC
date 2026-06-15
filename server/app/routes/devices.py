from fastapi import APIRouter

from server.app.schemas import DeviceRegisterRequest, DeviceStatusRequest
from server.app.storage import (
    DEVICES_DIR,
    generate_id,
    load_json,
    save_json,
    utc_now_iso,
    list_json_files,
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

    now = utc_now_iso()

    status_snapshot = {
        "status_id": generate_id("dst"),
        "received_at_utc": now,
        "device_id": payload.device_id,
        "firmware_version": payload.firmware_version,
        "model_version": payload.model_version,
        "battery_level": payload.battery_level,
        "free_memory_kb": payload.free_memory_kb,
        "signal_quality": payload.signal_quality,
        "extra": payload.extra,
    }

    device["last_seen_at_utc"] = now
    device["firmware_version"] = payload.firmware_version
    device["model_version"] = payload.model_version
    device["battery_level"] = payload.battery_level
    device["free_memory_kb"] = payload.free_memory_kb
    device["signal_quality"] = payload.signal_quality
    device["extra"] = payload.extra
    device["last_status_id"] = status_snapshot["status_id"]

    save_json(device_path, device)

    status_dir = DEVICES_DIR / "_status_history"
    status_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        status_dir / f'{status_snapshot["status_id"]}.json',
        status_snapshot,
    )

    return {
        "updated": True,
        "device_id": payload.device_id,
        "status_id": status_snapshot["status_id"],
    }


@router.get("")
def list_devices():
    devices = list_json_files(DEVICES_DIR)

    devices = [
        device for device in devices
        if device.get("device_id") is not None
    ]

    return {
        "count": len(devices),
        "devices": devices,
    }


@router.get("/status-history")
def list_status_history(device_id: str | None = None):
    status_dir = DEVICES_DIR / "_status_history"
    status_dir.mkdir(parents=True, exist_ok=True)

    statuses = list_json_files(status_dir)

    if device_id is not None:
        statuses = [
            status for status in statuses
            if status.get("device_id") == device_id
        ]

    return {
        "count": len(statuses),
        "statuses": statuses,
    }

@router.get("/{device_id}")
def get_device(device_id: str):
    device_path = DEVICES_DIR / f"{device_id}.json"
    if not device_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Dispositivo nao encontrado: {device_id}")
    return load_json(device_path)


@router.get("/{device_id}/compatibility")
def check_device_compatibility(device_id: str):
    device_path = DEVICES_DIR / f"{device_id}.json"
    if not device_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Dispositivo nao encontrado: {device_id}")

    device = load_json(device_path)
    device_type = (device.get("device_type") or "").lower()
    compatible = device_type in {"esp32", "esp32dev", "esp32-devkit"}

    return {
        "device_id": device_id,
        "device_type": device.get("device_type"),
        "compatible_with_current_firmware": compatible,
        "expected_target": "esp32",
        "reason": (
            "Dispositivo compativel com o firmware TFLite Micro atual."
            if compatible else
            "Firmware atual foi projetado para ESP32; use ESP32 para validacao embarcada."
        ),
    }
