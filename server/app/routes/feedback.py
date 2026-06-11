from fastapi import APIRouter, HTTPException
from server.app.schemas import FeedbackCreateRequest
from server.app.storage import FEEDBACK_DIR, generate_id, list_json_files, load_json, save_json, utc_now_iso, EVENTS_DIR

router = APIRouter()

@router.post("")
def create_feedback(payload: FeedbackCreateRequest):
    event_path = EVENTS_DIR / f"{payload.event_id}.json"

    if not event_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evento nao encontrado: {payload.event_id}",
        )

    event = load_json(event_path)

    feedback_id = generate_id("fbk")

    feedback = {
        "feedback_id": feedback_id,
        "event_id": payload.event_id,
        "created_at_utc": utc_now_iso(),
        "reviewer": payload.reviewer,
        "label": payload.label,
        "notes": payload.notes,
        "model_prediction": event.get("prediction"),
        "model_score": event.get("score"),
        "model_version": event.get("model_version"),
        "device_id": event.get("device_id"),
    }

    event["feedback_status"] = "reviewed"
    event["human_label"] = payload.label
    event["reviewed_at_utc"] = feedback["created_at_utc"]
    event["reviewer"] = payload.reviewer

    save_json(event_path, event)
    save_json(FEEDBACK_DIR / f"{feedback_id}.json", feedback)

    return {
        "saved": True,
        "feedback_id": feedback_id,
        "event_id": payload.event_id,
        "label": payload.label,
    }


@router.get("")
def list_feedback():
    from server.app.storage import list_json_files

    feedback_items = list_json_files(FEEDBACK_DIR)

    return {
        "count": len(feedback_items),
        "feedback": feedback_items,
    }