#Recebe os eventos do modelo,como anomalia, incerteza ou amostras normais
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from server.app.schemas import EventCreateRequest, EventResponse
from server.app.storage import EVENTS_DIR, generate_id, list_json_files, load_json, save_json, utc_now_iso

router = APIRouter()

def infer_priority(score: float, prediction: str) -> str:
    if prediction == "anomaly" and score >= 0.8:
        return "high_anomaly"

    if 0.4 <= score <= 0.6:
        return "uncertain"

    if prediction == "normal" and score <= 0.2:
        return "high_normal"

    return "low_priority"


@router.post("", response_model=EventResponse)
@router.post("/", response_model=EventResponse, include_in_schema=False)
def create_event(payload: EventCreateRequest):
    event_id = generate_id('evt')
    event_timestamp = payload.timestamp
    if event_timestamp is None:
        event_timestamp_str = datetime.now(timezone.utc).isoformat()
    elif isinstance(event_timestamp, str):
        event_timestamp_str = event_timestamp
    else:
        event_timestamp_str = event_timestamp.isoformat()

    event = {
        "event_id": event_id,
        "received_at_utc": utc_now_iso(),
        "event_timestamp": event_timestamp_str,
        "device_id": payload.device_id,
        "model_version": payload.model_version,
        "prediction": payload.prediction,
        "score": payload.score,
        "threshold": payload.threshold,
        "priority": infer_priority(payload.score, payload.prediction),
        "window_size": payload.window_size,
        "sampling_rate": payload.sampling_rate,
        "features": payload.features,
        "signal_window": payload.signal_window,
        "extra": payload.extra,
        "feedback_status": "pending_review",
        "human_label": None,
    }

    save_json(EVENTS_DIR / f"{event_id}.json", event)
    
    return EventResponse(**event)

@router.get('/')
def list_events(status: str | None = None):
    events = list_json_files(EVENTS_DIR)
    if status is not None:
        events = [
            event for event in events
            if event.get("feedback_status") == status
        ]
    return {
        'count': len(events),
        'events': events,
    }
    
@router.get('/{event_id}')
def get_event(event_id: str):
    path = EVENTS_DIR / f"{event_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Evento {event_id} nao encontrado")
    event = load_json(path)
    return event