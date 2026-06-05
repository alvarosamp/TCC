## profile le e valida o contrato do dominio
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any 
import yaml


def infer_window_size(x_shape: tuple[int, ...]) -> int:
    """Infere o tamanho da janela a partir da forma dos dados."""
    if len(x_shape) == 2:
        return int(x_shape[1])

    if len(x_shape) == 3 and x_shape[-1] == 1:
        return int(x_shape[1])

    if len(x_shape) == 3 and x_shape[1] == 1:
        return int(x_shape[2])

    raise ValueError(f"Formato de X nao suportado: {x_shape}")


@dataclass(frozen=True)
class PipelineProfile:
    '''Contrato da serie temporal
    Esse objeto conecta : dataset -> treino -> metrics -> export p tiny -> edge
    '''
    profile_name: str
    profile_version : str
    task : str
    domain : str
    description: str | None
    sampling_rate : float
    window_size : int
    window_seconds : float | None
    step_seconds : float | None
    overlap : float | None
    normal_label : int
    anomaly_label: int
    normal_name: str
    anomaly_name: str
    split_name:str
    primary_metric: str
    secondary_metrics: list[str]
    preprocessing: dict[str, Any]
    embedded: dict[str, Any]

    @classmethod
    def load_from_yaml(cls, path: Path) -> "PipelineProfile":
        """Carrega um profile a partir de um arquivo YAML."""
        return cls.from_yaml(path)
    
    @classmethod
    def from_yaml(cls,path: Path) -> PipelineProfile:
        """
        Carregau m profile a partir do yaml
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        required = [
            'profile_name', 'profile_version', 'task', 'domain',
            'sampling_rate', 'window_size', 'window_seconds', 'step_seconds',
            'overlap', 'normal_label', 'anomaly_label', 'normal_name',
            'anomaly_name', 'split_name', 'primary_metric', 'secondary_metrics',
            'preprocessing', 'embedded'
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required fields in profile: {missing}")

        data.setdefault('description', None)
        return cls(**data)
        
    @property
    def full_version(self) -> str:
        """Retorna o nome completo do perfil, combinando nome e versão."""
        return f"{self.profile_name}_v{self.profile_version}"
        
    def validate_window_size(self, x_shape: tuple[int, ...]) -> None:
        """Valida se a forma da janela é compatível com o perfil."""
        actual_window = infer_window_size(x_shape)
        if actual_window != self.window_size:
            raise ValueError(
                f"Janela incompativel com o profile {self.profile_name}. "
                f"Dataset tem {actual_window}, mas profile espera "
                f"{self.window_size}."
            )

    # Compatibilidade com nomes antigos no código
    def validate_window_shape(self, x_shape: tuple[int, ...]) -> None:
        self.validate_window_size(x_shape)
    
    def to_dict(self) -> dict[str, Any]:
        """Converte o perfil para um dicionário."""
        return {
            'profile_name': self.profile_name,
            'profile_version': self.profile_version,
            'task': self.task,
            'domain': self.domain,
            'description': self.description,
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'window_seconds': self.window_seconds,
            'step_seconds': self.step_seconds,
            'overlap': self.overlap,
            'normal_label': self.normal_label,
            'anomaly_label': self.anomaly_label,
            'normal_name': self.normal_name,
            'anomaly_name': self.anomaly_name,
            'split_name': self.split_name,
            'primary_metric': self.primary_metric,
            'secondary_metrics': self.secondary_metrics,
            'preprocessing': self.preprocessing,
            'embedded': self.embedded
        }

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
            
        return float(value)

