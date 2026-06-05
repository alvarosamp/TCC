"""
Exporta o melhor modelo edge para TFLite (float32, float16, int8).

Entrada : artefacts/reports/candidate_manifest.json
Saida   : artefacts/edge/<model>_float32.tflite
           artefacts/edge/<model>_float16.tflite
           artefacts/edge/<model>_int8.tflite
           artefacts/edge/<model>_int8.h

Comando:
  python -m src.export.export_tflite
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.core.settings import DATASET_FILE, EDGE_DIR, PROFILE_PATH, REPORTS_DIR, ensure_directories
from src.export.export_header import export_tflite_to_header

log = logging.getLogger("export_tflite")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

CANDIDATE_MANIFEST = REPORTS_DIR / "candidate_manifest.json"


# ============================================================
# IO
# ============================================================

def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# PREPARACAO DE ENTRADA
# ============================================================

def as_model_input(X: np.ndarray, model_name: str) -> np.ndarray:
    """dense_autoencoder usa (n, window_size); demais usam (n, window_size, 1)."""
    if model_name == "dense_autoencoder":
        return X.astype(np.float32, copy=False)
    return X[..., np.newaxis].astype(np.float32, copy=False)


def representative_dataset_generator(X_train: np.ndarray, model_name: str, n_samples: int = 512):
    """Dataset representativo para quantizacao int8."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(n_samples, len(X_train)), replace=False)
    X_rep = as_model_input(X_train[idx], model_name)

    def generator():
        for i in range(len(X_rep)):
            yield [X_rep[i : i + 1]]

    return generator


# ============================================================
# CONVERSAO
# ============================================================

def convert_float32(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_float16(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_int8(model: tf.keras.Model, X_train: np.ndarray, model_name: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator(
        X_train, model_name, n_samples=512
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def export_variant(variant_name: str, data: bytes, model_name: str) -> dict[str, Any]:
    out_path = EDGE_DIR / f"{model_name}_{variant_name}.tflite"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return {
        "variant": variant_name,
        "path": str(out_path),
        "size_bytes": len(data),
        "size_kb": round(len(data) / 1024, 2),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_directories()

    candidate = load_json(CANDIDATE_MANIFEST)
    model_name = candidate["model_name"]
    model_path = Path(candidate["model_path"])

    if not candidate.get("export_tflite", False):
        log.warning(
            f"Candidato '{model_name}' tem export_tflite=false — nada a exportar."
        )
        return

    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo do modelo nao encontrado: {model_path}")

    if model_path.suffix not in (".keras", ".h5"):
        raise ValueError(
            f"Formato nao suportado para TFLite: {model_path.suffix}. "
            "Use .keras ou .h5."
        )

    profile = PipelineProfile.from_yaml(PROFILE_PATH)
    X_train, _ = load_validated_split(DATASET_FILE, "train", profile)

    log.info(f"Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)
    log.info(f"Parametros: {model.count_params():,}")

    exports: dict[str, Any] = {}

    log.info("Convertendo float32...")
    exports["float32"] = export_variant("float32", convert_float32(model), model_name)

    log.info("Convertendo float16...")
    exports["float16"] = export_variant("float16", convert_float16(model), model_name)

    log.info("Convertendo int8...")
    exports["int8"] = export_variant(
        "int8", convert_int8(model, X_train, model_name), model_name
    )

    header_info = export_tflite_to_header(
        tflite_path=exports["int8"]["path"],
        header_path=EDGE_DIR / f"{model_name}_int8.h",
        array_name=f"{model_name}_int8_model_data",
    )

    export_manifest = {
        "model_name": model_name,
        "exports": exports,
        "header_info": header_info,
        "source_model_path": str(model_path),
        "profile": profile.to_dict(),
        "dataset": {"file": str(DATASET_FILE), "train_samples": len(X_train)},
        "threshold": candidate.get("threshold"),
        "metrics": candidate.get("metrics", {}),
    }

    out_manifest = EDGE_DIR / f"{model_name}_export_manifest.json"
    save_json(out_manifest, export_manifest)

    log.info("=" * 60)
    for name, info in exports.items():
        log.info(f"  {name:8s}  {info['size_kb']:.1f} KB  ->  {info['path']}")
    log.info(f"  header   ->  {header_info['header_path']}")
    log.info(f"  manifest ->  {out_manifest}")


if __name__ == "__main__":
    main()
