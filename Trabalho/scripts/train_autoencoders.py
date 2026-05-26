from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc_seismic.data import load_available_splits, normal_only
from tcc_seismic.io import write_json
from tcc_seismic.metrics import evaluate_with_validation_threshold
from tcc_seismic.paths import results_dir
from tcc_seismic.tf_models import (
    build_cnn1d_ae,
    build_dense_ae,
    build_lstm_ae,
    build_transformer_ae,
    require_tensorflow,
)


def reshape_for_model(model_name: str, X: np.ndarray, input_dim: int, timesteps: int) -> np.ndarray:
    if model_name == "cnn1d":
        return X.reshape(-1, input_dim, 1)
    if model_name == "lstm":
        return X.reshape(-1, timesteps, input_dim // timesteps)
    return X


def reconstruction_scores(model, model_name: str, X: np.ndarray, input_dim: int, timesteps: int) -> np.ndarray:
    X_in = reshape_for_model(model_name, X, input_dim, timesteps)
    rec = model.predict(X_in, batch_size=1024, verbose=0)
    rec = rec.reshape(-1, input_dim)
    return np.mean((X - rec) ** 2, axis=1)


def build_model(model_name: str, input_dim: int, args):
    if model_name == "dense":
        return build_dense_ae(input_dim, latent_dim=args.latent_dim, dropout=args.dropout)
    if model_name == "cnn1d":
        return build_cnn1d_ae(input_dim)
    if model_name == "lstm":
        return build_lstm_ae(input_dim, timesteps=args.timesteps)
    if model_name == "transformer":
        return build_transformer_ae(
            input_dim,
            patch_size=args.patch_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_one_split(split, args) -> dict:
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    input_dim = split.input_dim
    X_train_normal = normal_only(split.X_train, split.y_train)
    X_val_normal = normal_only(split.X_val, split.y_val)

    if len(X_train_normal) == 0:
        raise ValueError(f"{split.name}: no normal samples in train")
    if len(X_val_normal) == 0:
        raise ValueError(f"{split.name}: no normal samples in validation")

    model = build_model(args.model, input_dim, args)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        jit_compile=False,
    )

    X_train_in = reshape_for_model(args.model, X_train_normal, input_dim, args.timesteps)
    X_val_in = reshape_for_model(args.model, X_val_normal, input_dim, args.timesteps)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train_in,
        X_train_in,
        validation_data=(X_val_in, X_val_in),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_seconds = time.perf_counter() - start

    scores_val = reconstruction_scores(model, args.model, split.X_val, input_dim, args.timesteps)
    scores_test = reconstruction_scores(model, args.model, split.X_test, input_dim, args.timesteps)
    evaluation = evaluate_with_validation_threshold(
        split.y_val,
        scores_val,
        split.y_test,
        scores_test,
        strategy=args.threshold_strategy,
    )

    out_dir = results_dir("models_corrected", args.model, split.name)
    model_path = out_dir / f"{args.model}_{split.name}.keras"
    model.save(model_path)

    result = {
        "model": args.model,
        "split": split.name,
        "input_dim": input_dim,
        "train_samples_normal": int(len(X_train_normal)),
        "val_samples": int(len(split.X_val)),
        "test_samples": int(len(split.X_test)),
        "params_total": int(model.count_params()),
        "epochs_run": int(len(history.history["loss"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "train_seconds": float(train_seconds),
        "model_path": str(model_path),
        "evaluation": evaluation,
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
    }
    write_json(out_dir / "results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train corrected autoencoder baselines for seismic event detection."
    )
    parser.add_argument(
        "--model",
        choices=["dense", "cnn1d", "lstm", "transformer"],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "legacy_v2", "legacy_v1", "v3"],
        default="auto",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["best_f1", "percentile_normal", "target_recall"],
        default="best_f1",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_available_splits(args.dataset)
    all_results = [train_one_split(split, args) for split in splits]
    out_dir = results_dir("models_corrected", args.model)
    write_json(out_dir / "summary.json", {"model": args.model, "results": all_results})
    print(json.dumps({"model": args.model, "results_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

