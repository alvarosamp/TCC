"""
Cria a referencia de drift a partir do conjunto de treino.

Entrada:
  dataset configurado em config/configs/config.yaml

Saida:
  artefacts/monitoring/drift_reference.json

Ideia:
  O conjunto de treino representa a distribuicao conhecida.
  Quando dados novos chegarem, vamos comparar suas features com
  essa referencia para detectar data drift.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.core.settings import ARTEFACTS_DIR, DATASET_FILE, PROFILE_PATH
from src.features.statistical_features import extract_statistical_features, feature_names

MONITORING_DIR = ARTEFACTS_DIR / "monitoring"
DRIFT_REFERENCE = MONITORING_DIR / "drift_reference.json"

def save_json(path: Path,  payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(payload, f, indent = 4)
        
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def summarize_features(features: np.ndarray, names: list[str]) -> dict[str,dict[str, float]]:
    """
    Calcula estatisticas por feature.

    Cada feature recebe:
      mean, std, min, max, p05, p25, p50, p75, p95
    """
    if features.ndim != 2:
        raise ValueError(f"features deve ser 2D, recebido: {features.shape}")

    summary: dict[str, dict[str, float]] = {}

    for idx, name in enumerate(names):
        values = features[:, idx].astype(np.float64)

        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }

    return summary


def build_reference() -> dict[str, Any]:
    '''
    Carrega o treino, extrai features e cria referencia de drift.
    '''
    profile = PipelineProfile.from_yaml(PROFILE_PATH)
    X_train, y_train = load_validated_split(DATASET_FILE, 'train', profile)
    names = feature_names()
    features_train = extract_statistical_features(
        X_train,
        sample_rate = profile.sampling_rate,
    )
    if features_train.shape[1] != len(names):
        raise ValueError(
            "Numero de features extraidas nao bate com numero de nomes: "
            f"{features_train.shape[1]} != {len(names)}"
        )
    label_counts = {
        'normal' : int((y_train == profile.normal_label).sum()),
        'anomaly' : int((y_train != profile.normal_label).sum()),
        'total' : int(len(y_train)),
    }
    return {
    "schema_version": "1.0.0",
    "created_at_utc": utc_now_iso(),
    "dataset": str(DATASET_FILE),
    "profile": profile.to_dict(),
    "split": "train",
    "n_samples": int(len(X_train)),
    "window_shape": list(X_train.shape[1:]),
    "feature_count": int(features_train.shape[1]),
    "feature_names": names,
    "label_counts": label_counts,
    "reference_stats": summarize_features(
        features=features_train,
        names=names,
    ),
}
    
def main() -> None:
    reference = build_reference()

    save_json(DRIFT_REFERENCE, reference)

    print("=" * 80)
    print("REFERENCIA DE DRIFT GERADA")
    print("=" * 80)
    print(f"Dataset:  {reference['dataset']}")
    print(f"Split:    {reference['split']}")
    print(f"Amostras: {reference['n_samples']}")
    print(f"Features: {reference['feature_count']}")
    print(f"Saida:    {DRIFT_REFERENCE}")


if __name__ == "__main__":
    main()
    
