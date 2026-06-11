from datetime import datetime
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field

PredictionLabel = Literal['normal','anomaly', 'uncertain']
FeedbackLabel = Literal[
    'confirmed_anomaly',
    'confirmed_normal',
    'false_positive',
    'false_negative',
    'uncertain',
    'discarted',
]

class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., examples = ['esp32_001'])
    device_type:str = Field(default = 'esp32')
    location: Optional[str] = Field(default = None, examples = ['campo_teste'])
    firmware_version: Optional[str] = Field(default = None, examples = ['v1.0.0'])
    model_version: Optional[str] = Field(default = None, examples = ['seismic_edge_v1'])
    
    
class DeviceStatusRequest(BaseModel):
    device_id: str
    firmware_version: Optional[str] = None
    model_version: Optional[str] = None
    battery_level: Optional[float] = None
    free_memory_kb: Optional[int] = None
    signal_quality: Optional[float] = None
    extra: Optional[dict[str, Any]] = None
    
class EventCreateRequest(BaseModel):
    device_id: str
    timestamp: Optional[datetime] = None
    model_version: str
    prediction: PredictionLabel
    score: float = Field(..., ge=0.0, le=1.0)
    threshold: Optional[float] = None
    window_size: Optional[int] = None
    sampling_rate: Optional[float] = None
    features: Optional[dict[str, Any]] = None
    signal_window: Optional[list[float]] = None
    extra: Optional[dict[str, Any]] = None


class EventResponse(BaseModel):
    event_id: str
    status: str
    saved: bool


class FeedbackCreateRequest(BaseModel):
    event_id: str
    reviewer: str = Field(default="operator")
    label: FeedbackLabel
    notes: Optional[str] = None


class OtaReportRequest(BaseModel):
    device_id: str
    previous_model_version: Optional[str] = None
    new_model_version: str
    status: Literal["success", "failed"]
    message: Optional[str] = None