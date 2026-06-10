"""
Avalia os modelos TFLite exportados no conjunto de teste.

Compara float32, float16 e int8 em termos de metricas de classificacao
e tamanho, permitindo quantificar o impacto da quantizacao.

Entrada:
  artefacts/edge/<model_name>_export_manifest.json

Saida:
  artefacts/reports/tflite_evaluation.json

Comando:
  python -m src.export.evaluate_tflite
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.core.settings import DATASET_FILE, EDGE_DIR, PROFILE_PATH, REPORTS_DIR
from src.training.evaluate import evaluate_scores

log = logging.getLogger("evaluate_tflite")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

CANDIDATE_MANIFEST = REPORTS_DIR / "candidate_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def as_model_input(X: np.ndarray, model_name: str) -> np.ndarray:
    if model_name == "dense_autoencoder":
        return X.astype(np.float32, copy=False)
    return X[..., np.newaxis].astype(np.float32, copy=False)


def run_tflite_inference(
    tflite_path: Path,
    X: np.ndarray,
) -> np.ndarray:
    """
    Roda inferencia com TFLite interpreter e retorna scores float32.

    Lida com float32, float16 (input float32) e int8 (quantiza/dequantiza).
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    is_int8 = input_details["dtype"] == np.int8
    if is_int8:
        in_scale, in_zero = input_details["quantization"]
        out_scale, out_zero = output_details["quantization"]

    scores = np.empty(len(X), dtype=np.float32)

    for i, sample in enumerate(X):
        inp = sample[np.newaxis]
        if is_int8:
            inp = np.clip(
                np.round(inp / in_scale + in_zero), -128, 127
            ).astype(np.int8)

        interpreter.set_tensor(input_details["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])

        if is_int8:
            out = (out.astype(np.float32) - out_zero) * out_scale

        scores[i] = float(out.ravel()[0])

    return scores


def evaluate_variant(
    variant: str,
    tflite_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    size_kb: float,
) -> dict[str, Any]:
    log.info(f"Avaliando {variant} ({size_kb:.1f} KB)...")

    t0 = time.perf_counter()
    scores = run_tflite_inference(tflite_path, X_test)
    elapsed = time.perf_counter() - t0

    metrics = evaluate_scores(y_true=y_test, scores=scores, threshold=threshold)

    return {
        "variant": variant,
        "size_kb": size_kb,
        "inference_total_s": round(elapsed, 4),
        "inference_per_sample_ms": round(elapsed / len(X_test) * 1000, 4),
        "auc_pr": metrics["auc_pr"],
        "auc_roc": metrics["auc_roc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "threshold": threshold,
        "confusion_matrix": metrics["confusion_matrix"],
    }


def main() -> None:
    candidate = load_json(CANDIDATE_MANIFEST)
    model_name = candidate["model_name"]
    threshold = float(candidate["threshold"])

    export_manifest = load_json(EDGE_DIR / f"{model_name}_export_manifest.json")
    exports = export_manifest["exports"]

    profile = PipelineProfile.from_yaml(PROFILE_PATH)
    X_test, y_test = load_validated_split(DATASET_FILE, "test", profile)
    X_test = as_model_input(X_test, model_name)

    log.info(f"Modelo: {model_name} | Threshold: {threshold:.4f} | Amostras: {len(X_test)}")

    results: list[dict[str, Any]] = []
    for variant_name, info in exports.items():
        result = evaluate_variant(
            variant=variant_name,
            tflite_path=Path(info["path"]),
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
            size_kb=info["size_kb"],
        )
        results.append(result)

    report = {
        "model_name": model_name,
        "threshold": threshold,
        "test_samples": len(X_test),
        "variants": results,
    }

    out_path = REPORTS_DIR / "tflite_evaluation.json"
    save_json(out_path, report)

    log.info("=" * 60)
    log.info(f"{'Variante':<12} {'Size KB':>8} {'AUC-PR':>8} {'F1':>8} {'ms/sample':>10}")
    log.info("-" * 60)
    for r in results:
        log.info(
            f"{r['variant']:<12} {r['size_kb']:>8.1f} "
            f"{r['auc_pr']:>8.4f} {r['f1']:>8.4f} "
            f"{r['inference_per_sample_ms']:>10.3f}"
        )
    log.info(f"Relatorio: {out_path}")


if __name__ == "__main__":
    main()
