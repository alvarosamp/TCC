from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an edge-oriented binary time-series classifier using a "
            "versioned pipeline profile."
        )
    )
    parser.add_argument("--profile", required=True, help="Path to pipeline profile JSON.")
    parser.add_argument("--dataset", required=True, help="Path to split NPZ dataset.")
    parser.add_argument("--model", choices=["tiny_tcn", "tiny_cnn"], default="tiny_tcn")
    parser.add_argument("--output-dir", default="artefacts/runs/generic_classifier")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--export-tflite",
        choices=["none", "float32", "float16", "int8", "all"],
        default="none",
        help="Export TensorFlow Lite variants after training.",
    )
    parser.add_argument(
        "--export-header",
        action="store_true",
        help="Also export TFLite files as C headers for PlatformIO/TFLM.",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=512,
        help="Number of train windows used for int8 representative dataset.",
    )
    return parser.parse_args()


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def as_model_input(x: np.ndarray) -> np.ndarray:
    return x[..., np.newaxis].astype(np.float32, copy=False)


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    labels, counts = np.unique(y.astype(int), return_counts=True)
    total = float(len(y))
    n_classes = float(len(labels))
    return {int(label): total / (n_classes * float(count)) for label, count in zip(labels, counts)}


def predict_scores(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    scores = model.predict(as_model_input(x), batch_size=batch_size, verbose=0)
    return np.asarray(scores).reshape(-1).astype(np.float32)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def representative_dataset(x_train: np.ndarray, n_samples: int):
    x = as_model_input(x_train[:n_samples])

    def generator():
        for i in range(len(x)):
            yield [x[i : i + 1]]

    return generator


def export_tflite_variants(
    tf: Any,
    model: Any,
    out_dir: Path,
    export_mode: str,
    x_train: np.ndarray,
    representative_samples: int,
) -> dict[str, str]:
    if export_mode == "none":
        return {}

    variants = ["float32", "float16", "int8"] if export_mode == "all" else [export_mode]
    exported: dict[str, str] = {}

    for variant in variants:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if variant == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif variant == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset(
                x_train,
                representative_samples,
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        path = out_dir / f"model_{variant}.tflite"
        path.write_bytes(tflite_model)
        exported[variant] = str(path)

    return exported


def c_identifier(name: str) -> str:
    clean = []
    for char in name:
        clean.append(char if char.isalnum() else "_")
    return "".join(clean).strip("_")


def export_c_header(tflite_path: Path, header_path: Path, array_name: str) -> None:
    data = tflite_path.read_bytes()
    values = ", ".join(str(byte) for byte in data)
    guard = c_identifier(header_path.name).upper()
    text = (
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        "#include <cstdint>\n\n"
        f"alignas(16) const unsigned char {array_name}[] = {{\n"
        f"{values}\n"
        "};\n"
        f"const unsigned int {array_name}_len = {len(data)};\n\n"
        f"#endif  // {guard}\n"
    )
    header_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    from tcc_pipeline.dataset import load_npz_split, summarize_split
    from tcc_pipeline.metrics import evaluate_scores
    from tcc_pipeline.models import build_classifier, require_tensorflow
    from tcc_pipeline.profile import PipelineProfile

    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)

    profile = PipelineProfile.from_json(args.profile)
    dataset_path = Path(args.dataset)
    run_name = args.run_name or f"{profile.profile_name}_{args.model}_{now_slug()}"
    out_dir = (ROOT / args.output_dir / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_npz_split(dataset_path, "train", profile)
    x_val, y_val = load_npz_split(dataset_path, "val", profile)
    x_test, y_test = load_npz_split(dataset_path, "test", profile)

    model = build_classifier(
        args.model,
        window_size=profile.window_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        as_model_input(x_train),
        y_train,
        validation_data=(as_model_input(x_val), y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_from_labels(y_train),
        callbacks=callbacks,
        verbose=2,
    )

    scores_val = predict_scores(model, x_val, args.batch_size)
    val_metrics = evaluate_scores(y_val, scores_val)
    threshold = float(val_metrics["threshold"])

    scores_test = predict_scores(model, x_test, args.batch_size)
    test_metrics = evaluate_scores(y_test, scores_test, threshold=threshold)

    model_path = out_dir / f"{args.model}.keras"
    model.save(model_path)

    exported_tflite = export_tflite_variants(
        tf=tf,
        model=model,
        out_dir=out_dir,
        export_mode=args.export_tflite,
        x_train=x_train,
        representative_samples=args.representative_samples,
    )

    exported_headers: dict[str, str] = {}
    if args.export_header:
        for variant, path_str in exported_tflite.items():
            tflite_path = Path(path_str)
            header_path = out_dir / f"model_{variant}.h"
            export_c_header(
                tflite_path,
                header_path,
                array_name=f"g_{c_identifier(args.model)}_{variant}_model",
            )
            exported_headers[variant] = str(header_path)

    manifest = {
        "run_name": run_name,
        "created_at_utc": now_slug(),
        "profile": profile.to_dict(),
        "dataset": str(dataset_path),
        "model": {
            "name": args.model,
            "keras_path": str(model_path),
            "parameter_count": int(model.count_params()),
            "threshold": threshold,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_ran": len(history.history.get("loss", [])),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "seed": args.seed,
            "class_weight": class_weight_from_labels(y_train),
            "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        },
        "dataset_summary": {
            "train": summarize_split(x_train, y_train, profile),
            "val": summarize_split(x_val, y_val, profile),
            "test": summarize_split(x_test, y_test, profile),
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "exports": {
            "tflite": exported_tflite,
            "headers": exported_headers,
        },
        "notes": {
            "generic_contract": "The script depends on profile + NPZ split keys, not on seismic-specific code.",
            "edge_contract": "Use the same profile/preprocessing assumptions when exporting data to ESP32.",
        },
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(
        out_dir / "metrics.json",
        {
            "threshold_from_val": threshold,
            "val": val_metrics,
            "test": test_metrics,
        },
    )

    print(
        json.dumps(
            {
                "run_dir": str(out_dir),
                "model": args.model,
                "threshold": threshold,
                "test_auc_pr": test_metrics["auc_pr"],
                "test_f1": test_metrics["f1"],
                "exports": manifest["exports"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
