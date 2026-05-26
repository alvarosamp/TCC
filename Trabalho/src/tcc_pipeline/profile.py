from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineProfile:
    """Contrato versionado entre dataset, treino, edge e MLOps."""

    profile_name: str
    profile_version: str
    task: str
    domain: str
    description: str
    sampling_rate: float
    window_size: int
    window_seconds: float | None
    step_seconds: float | None
    overlap: float | None
    normal_label: int
    anomaly_label: int
    normal_name: str
    anomaly_name: str
    split_name: str
    primary_metric: str
    secondary_metrics: list[str]
    preprocessing: dict[str, Any]
    embedded: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineProfile":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        required = [
            "profile_name",
            "profile_version",
            "task",
            "domain",
            "sampling_rate",
            "window_size",
            "normal_label",
            "anomaly_label",
            "normal_name",
            "anomaly_name",
            "split_name",
            "primary_metric",
            "secondary_metrics",
            "preprocessing",
            "embedded",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Profile invalido. Campos ausentes: {missing}")

        return cls(
            profile_name=str(data["profile_name"]),
            profile_version=str(data["profile_version"]),
            task=str(data["task"]),
            domain=str(data["domain"]),
            description=str(data.get("description", "")),
            sampling_rate=float(data["sampling_rate"]),
            window_size=int(data["window_size"]),
            window_seconds=_optional_float(data.get("window_seconds")),
            step_seconds=_optional_float(data.get("step_seconds")),
            overlap=_optional_float(data.get("overlap")),
            normal_label=int(data["normal_label"]),
            anomaly_label=int(data["anomaly_label"]),
            normal_name=str(data["normal_name"]),
            anomaly_name=str(data["anomaly_name"]),
            split_name=str(data["split_name"]),
            primary_metric=str(data["primary_metric"]),
            secondary_metrics=[str(v) for v in data["secondary_metrics"]],
            preprocessing=dict(data["preprocessing"]),
            embedded=dict(data["embedded"]),
        )

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

    def validate_window_shape(self, x_shape: tuple[int, ...]) -> None:
        actual_window = infer_window_size(x_shape)
        if actual_window != self.window_size:
            raise ValueError(
                f"Janela incompativel com {self.profile_name}: "
                f"dataset={actual_window}, profile={self.window_size}"
            )

    @property
    def full_version(self) -> str:
        return f"{self.profile_name}:{self.profile_version}"


def infer_window_size(x_shape: tuple[int, ...]) -> int:
    if len(x_shape) == 2:
        return int(x_shape[1])
    if len(x_shape) == 3 and x_shape[-1] == 1:
        return int(x_shape[1])
    if len(x_shape) == 3 and x_shape[1] == 1:
        return int(x_shape[2])
    raise ValueError(f"Formato de X nao suportado: {x_shape}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
