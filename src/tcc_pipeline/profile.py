from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Trabalho.src.tcc_pipeline.profile import infer_window_size

@dataclass
class PipelineProfile:
    """
    Contrato versionado entre dataset, edge e mlops
    """
    profile_name: str
    profile_version : str
    task: str
    domain:str
    description: str
    sampling_rate: float
    window_size: int
    window_seconds : float | None
    step_seconds: float | None
    overlap: float | None
    normal_label: int
    anomaly_label: int
    normal_name:str
    anomaly_name:str
    split_name: str
    primary_metric: str
    secondary_metric: list[str]
    preprocessing: dict[str, Any]
    embedded: dict[str, Any]
    
    
    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineProfile":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        
        required = [
            'profile_name',
            'profile_version',
            'task',
            "domain",
            'sampling_rate',
            'window_size',
            'normal_label',
            'anomaly_label',
            'normal_name',
            'anomaly_name',
            'split_name',
            'primary_metric',
            "secondary_metric",
            'preprocessing',
            'embedded'
        ]
        
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required fields in profile: {missing}")
        
        return cls(
            profile_name=data['profile_name'],
            profile_version=data['profile_version'],
            task=data['task'],
            domain=data['domain'],
            description=data['description'],
            sampling_rate=data['sampling_rate'],
            window_size=data['window_size'],
            window_seconds=data.get('window_seconds'),
            step_seconds=data.get('step_seconds'),
            overlap=data.get('overlap'),
            normal_label=data['normal_label'],
            anomaly_label=data['anomaly_label'],
            normal_name=data['normal_name'],
            anomaly_name=data['anomaly_name'],
            split_name=data['split_name'],
            primary_metric=data['primary_metric'],
            secondary_metric=data['secondary_metric'],
            preprocessing=data['preprocessing'],
            embedded=data['embedded']
        )
        
    def validate_window_shape(self, X_shape: tuple[int, ...]) -> None:
        actual_window = infer_window_size(X_shape)
        if actual_window != self.window_size:
            raise ValueError(f"Expected window size {self.window_size} does not match actual window size {actual_window}")
        
    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "task": self.task,
            "domain": self.domain,
            "description": self.description,
            "sampling_rate": self.sampling_rate,
            "window_size": self.window_size,
            "window_seconds": self.window_seconds,
            "step_seconds": self.step_seconds,
            "overlap": self.overlap,
            "normal_label": self.normal_label,
            "anomaly_label": self.anomaly_label,
            "normal_name": self.normal_name,
            "anomaly_name": self.anomaly_name,
            "split_name": self.split_name,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "preprocessing": self.preprocessing,
            "embedded": self.embedded,
        }
    
    @property
    def full_version(self) -> str:
        return f"{self.profile_name}_v{self.profile_version}"
    
    def infer_window_size(X_shape: tuple[int, ...]) -> int:
        if len(X_shape) == 2:
            return int(X_shape[1])
        if len(X_shape) == 3 and X_shape[-1] == 1:
            return int(X_shape[1])
        if len(X_shape) == 3 and X_shape[1] == 1:
            return int(X_shape[2])
        raise ValueError(f"Cannot infer window size from shape {X_shape}")
    
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)