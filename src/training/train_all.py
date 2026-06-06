"""
Pipeline completo de treinamento.

Suporta familias:
  classical_supervised   : logistic_regression, random_forest, extra_trees
  classical_unsupervised : isolation_forest
  neural_classifier      : tiny_cnn, tiny_tcn, lstm_classifier
  autoencoder            : dense_autoencoder, cnn_autoencoder

Fluxo:
  1. carrega dataset validado
  2. le config/model/models.yaml (ou models_smoke.yaml via env TCC_MODELS_CFG)
  3. treina enabled=true
  4. Optuna para modelos classicos quando habilitado
  5. avalia em validacao e teste
  6. registra no MLflow
  7. gera tabela comparativa
  8. salva candidate_manifest.json com melhor modelo edge

Comando:
  python -m src.training.train_all
  TCC_MODELS_CFG=config/model/models_smoke.yaml python -m src.training.train_all
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import tensorflow as tf
import yaml

from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.core.settings import (
    DATASET_FILE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROFILE_PATH,
    REPORTS_DIR,
    ROOT_DIR,
    SEED,
    ensure_directories,
)
from src.features.statistical_features import (
    extract_statistical_features,
    feature_names,
)
from src.training.evaluate import (
    evaluate_scores,
    evaluate_with_validation_threshold,
)
from src.training.hpo import suggest_params
from src.training.model_registry import build_model
from src.training.neural_models import build_neural_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger("train_all")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

_default_models_cfg = ROOT_DIR / "config" / "model" / "models.yaml"


# ============================================================
# IO
# ============================================================

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# PREPARACAO DE ENTRADA
# ============================================================

def as_conv_input(X: np.ndarray) -> np.ndarray:
    """(n, window_size) -> (n, window_size, 1) para Conv1D/LSTM."""
    return X[..., np.newaxis].astype(np.float32, copy=False)


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    labels, counts = np.unique(y.astype(int), return_counts=True)
    total = float(len(y))
    n_classes = float(len(labels))
    return {
        int(label): total / (n_classes * float(count))
        for label, count in zip(labels, counts)
    }


# ============================================================
# SCORES
# ============================================================

def supervised_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float32)
    raise TypeError(f"{type(model).__name__} nao possui score supervisionado.")


def unsupervised_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "score_samples"):
        return (-model.score_samples(X)).astype(np.float32)
    if hasattr(model, "decision_function"):
        return (-model.decision_function(X)).astype(np.float32)
    raise TypeError(f"{type(model).__name__} nao possui score nao supervisionado.")


# ============================================================
# CLASSICOS COM OPTUNA
# ============================================================

def train_classical_once(
    model_name: str,
    family: str,
    params: dict[str, Any],
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_val_f: np.ndarray,
    y_val: np.ndarray,
    profile: PipelineProfile,
) -> dict[str, Any]:
    model = build_model(model_name=model_name, params=params, seed=SEED)

    if family == "classical_supervised":
        model.fit(X_train_f, y_train)
        scores_val = supervised_scores(model, X_val_f)
    elif family == "classical_unsupervised":
        normal_mask = y_train == profile.normal_label
        model.fit(X_train_f[normal_mask])
        scores_val = unsupervised_scores(model, X_val_f)
    else:
        raise ValueError(f"Familia classica nao suportada: {family}")

    val_metrics = evaluate_scores(y_true=y_val, scores=scores_val, threshold=None)
    return {"model": model, "scores_val": scores_val, "val_metrics": val_metrics}


def tune_classical_model(
    model_name: str,
    model_cfg: dict[str, Any],
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_val_f: np.ndarray,
    y_val: np.ndarray,
    profile: PipelineProfile,
    selection_metric: str,
) -> dict[str, Any]:
    family = model_cfg["family"]
    base_params = dict(model_cfg.get("params", {}))
    tune_cfg = dict(model_cfg.get("tune", {}))
    search_space = dict(tune_cfg.get("search_space", {}))

    if not tune_cfg.get("enabled", False) or not search_space:
        return {
            "used_optuna": False,
            "best_params": base_params,
            "best_value": None,
            "n_trials": 0,
        }

    n_trials = int(tune_cfg.get("n_trials", 20))

    def objective(trial: optuna.Trial) -> float:
        trial_params = dict(base_params)
        trial_params.update(suggest_params(trial=trial, search_space=search_space))
        result = train_classical_once(
            model_name=model_name,
            family=family,
            params=trial_params,
            X_train_f=X_train_f,
            y_train=y_train,
            X_val_f=X_val_f,
            y_val=y_val,
            profile=profile,
        )
        return float(result["val_metrics"][selection_metric])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(base_params)
    best_params.update(study.best_params)
    return {
        "used_optuna": True,
        "best_params": best_params,
        "best_value": float(study.best_value),
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
    }


def train_classical_final(
    model_name: str,
    model_cfg: dict[str, Any],
    params: dict[str, Any],
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_val_f: np.ndarray,
    y_val: np.ndarray,
    X_test_f: np.ndarray,
    y_test: np.ndarray,
    profile: PipelineProfile,
) -> dict[str, Any]:
    family = model_cfg["family"]
    model = build_model(model_name=model_name, params=params, seed=SEED)

    if family == "classical_supervised":
        model.fit(X_train_f, y_train)
        scores_val = supervised_scores(model, X_val_f)
        scores_test = supervised_scores(model, X_test_f)
    elif family == "classical_unsupervised":
        normal_mask = y_train == profile.normal_label
        model.fit(X_train_f[normal_mask])
        scores_val = unsupervised_scores(model, X_val_f)
        scores_test = unsupervised_scores(model, X_test_f)
    else:
        raise ValueError(f"Familia classica nao suportada: {family}")

    evaluation = evaluate_with_validation_threshold(
        y_val=y_val,
        scores_val=scores_val,
        y_test=y_test,
        scores_test=scores_test,
    )

    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    return {"model_path": str(model_path), "evaluation": evaluation}


# ============================================================
# NEURAL CLASSIFIER
# ============================================================

def train_neural_classifier(
    model_name: str,
    model_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    profile: PipelineProfile,
) -> dict[str, Any]:
    params = dict(model_cfg.get("params", {}))
    epochs = int(params.get("epochs", 40))
    batch_size = int(params.get("batch_size", 256))
    patience = int(params.get("patience", 8))

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_neural_model(
        model_name=model_name,
        window_size=profile.window_size,
        params=params,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        as_conv_input(X_train),
        y_train,
        validation_data=(as_conv_input(X_val), y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_from_labels(y_train),
        callbacks=callbacks,
        verbose=2,
    )

    scores_val = model.predict(
        as_conv_input(X_val), batch_size=batch_size, verbose=0
    ).reshape(-1)
    scores_test = model.predict(
        as_conv_input(X_test), batch_size=batch_size, verbose=0
    ).reshape(-1)

    evaluation = evaluate_with_validation_threshold(
        y_val=y_val,
        scores_val=scores_val,
        y_test=y_test,
        scores_test=scores_test,
    )

    model_path = MODELS_DIR / f"{model_name}.keras"
    model.save(model_path)

    return {
        "model_path": str(model_path),
        "evaluation": evaluation,
        "history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        "parameter_count": int(model.count_params()),
    }


# ============================================================
# AUTOENCODER
# ============================================================

def reconstruction_scores(
    model: tf.keras.Model,
    X: np.ndarray,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    if model_name == "dense_autoencoder":
        X_input = X.astype(np.float32, copy=False)
        reconstructed = model.predict(X_input, batch_size=batch_size, verbose=0)
        errors = np.mean((X_input - reconstructed) ** 2, axis=1)
    else:
        X_input = as_conv_input(X)
        reconstructed = model.predict(X_input, batch_size=batch_size, verbose=0)
        errors = np.mean((X_input - reconstructed) ** 2, axis=(1, 2))
    return errors.astype(np.float32)


def train_autoencoder(
    model_name: str,
    model_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    profile: PipelineProfile,
) -> dict[str, Any]:
    params = dict(model_cfg.get("params", {}))
    epochs = int(params.get("epochs", 40))
    batch_size = int(params.get("batch_size", 256))
    patience = int(params.get("patience", 8))

    normal_mask = y_train == profile.normal_label
    X_train_normal = X_train[normal_mask]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_neural_model(
        model_name=model_name,
        window_size=profile.window_size,
        params=params,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-5,
        ),
    ]

    if model_name == "dense_autoencoder":
        X_train_input = X_train_normal.astype(np.float32, copy=False)
        X_val_input = X_val.astype(np.float32, copy=False)
    else:
        X_train_input = as_conv_input(X_train_normal)
        X_val_input = as_conv_input(X_val)

    history = model.fit(
        X_train_input,
        X_train_input,
        validation_data=(X_val_input, X_val_input),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    scores_val = reconstruction_scores(model, X_val, model_name, batch_size)
    scores_test = reconstruction_scores(model, X_test, model_name, batch_size)

    evaluation = evaluate_with_validation_threshold(
        y_val=y_val,
        scores_val=scores_val,
        y_test=y_test,
        scores_test=scores_test,
    )

    model_path = MODELS_DIR / f"{model_name}.keras"
    model.save(model_path)

    return {
        "model_path": str(model_path),
        "evaluation": evaluation,
        "history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        "parameter_count": int(model.count_params()),
    }


# ============================================================
# MLFLOW E RELATORIOS
# ============================================================

def log_result_to_mlflow(result: dict[str, Any]) -> None:
    model_name = result["model_name"]
    evaluation = result["evaluation"]
    metrics_path = REPORTS_DIR / f"{model_name}_metrics.json"
    save_json(metrics_path, result)

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("family", result["family"])
    mlflow.log_param(
        "profile",
        result["profile"]["profile_name"] + ":" + result["profile"]["profile_version"],
    )
    mlflow.log_param("dataset", result["dataset"])
    mlflow.log_param("edge_candidate", result["edge_candidate"])
    mlflow.log_param("export_tflite", result["export_tflite"])
    mlflow.log_param("priority", result["priority"])
    mlflow.log_param("used_optuna", result["hpo"]["used_optuna"])

    if "parameter_count" in result:
        mlflow.log_metric("parameter_count", result["parameter_count"])

    for key, value in result["params"].items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            mlflow.log_param(f"param_{key}", value)

    mlflow.log_metric("val_auc_pr", evaluation["val"]["auc_pr"])
    mlflow.log_metric("val_f1", evaluation["val"]["f1"])
    mlflow.log_metric("test_auc_pr", evaluation["test"]["auc_pr"])
    mlflow.log_metric("test_f1", evaluation["test"]["f1"])
    mlflow.log_metric("threshold_from_val", evaluation["threshold_from_val"])

    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(result["model_path"])


def metric_value(result: dict[str, Any], split: str, metric: str) -> float:
    return float(result["evaluation"][split][metric])


def select_best_edge_candidate(
    results: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Seleciona o melhor modelo marcado como edge_candidate=True."""
    metric = selection_cfg.get("metric", "auc_pr")
    split = selection_cfg.get("split", "test")
    mode = selection_cfg.get("mode", "maximize")

    edge = [r for r in results if r.get("edge_candidate", False)]
    if not edge:
        return None

    reverse = mode == "maximize"
    return sorted(edge, key=lambda r: metric_value(r, split, metric), reverse=reverse)[0]


def select_best_model(
    results: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Seleciona o melhor modelo geral (para comparacao)."""
    if not results:
        raise ValueError("Nenhum resultado disponivel para selecao.")

    metric = selection_cfg.get("metric", "auc_pr")
    split = selection_cfg.get("split", "test")
    mode = selection_cfg.get("mode", "maximize")
    reverse = mode == "maximize"

    return sorted(
        results,
        key=lambda r: metric_value(r, split, metric),
        reverse=reverse,
    )[0]


def _fp_per_hour(metrics: dict[str, Any], profile: dict[str, Any]) -> float | None:
    """Estima falsos positivos por hora para uma avaliacao por janelas.

    O denominador e o tempo coberto pelas decisoes avaliadas:
      n_janelas * step_seconds.

    Isso permite comparar modelos pensando em operacao real. Um modelo pode ter
    AUC-PR alta, mas ainda assim incomodar se gerar muitos alarmes falsos por
    hora.
    """
    step_seconds = profile.get("step_seconds")
    if step_seconds is None or float(step_seconds) <= 0:
        return None

    cm = metrics.get("confusion_matrix", {})
    n_windows = int(cm.get("tn", 0)) + int(cm.get("fp", 0)) + int(cm.get("fn", 0)) + int(cm.get("tp", 0))
    if n_windows <= 0:
        return None

    fp = int(cm.get("fp", 0))
    hours = (n_windows * float(step_seconds)) / 3600.0
    return float(fp / hours) if hours > 0 else None


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    val = result["evaluation"]["val"]
    test = result["evaluation"]["test"]
    profile = result["profile"]
    return {
        "model_name": result["model_name"],
        "family": result["family"],
        "priority": result["priority"],
        "edge_candidate": result["edge_candidate"],
        "export_tflite": result["export_tflite"],
        "used_optuna": result["hpo"]["used_optuna"],
        "parameter_count": result.get("parameter_count", ""),
        "val_auc_pr": val["auc_pr"],
        "val_auc_roc": val["auc_roc"],
        "val_f1": val["f1"],
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_fp_per_hour": _fp_per_hour(val, profile),
        "test_auc_pr": test["auc_pr"],
        "test_auc_roc": test["auc_roc"],
        "test_f1": test["f1"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_fp_per_hour": _fp_per_hour(test, profile),
        "threshold_from_val": result["evaluation"]["threshold_from_val"],
        "model_path": result["model_path"],
    }


def save_comparison_reports(
    results: list[dict[str, Any]],
    best_overall: dict[str, Any],
    best_edge: dict[str, Any] | None,
    selection_cfg: dict[str, Any],
) -> None:
    rows = [_metric_row(r) for r in results]

    csv_path = REPORTS_DIR / "model_comparison.csv"
    md_path = REPORTS_DIR / "model_comparison.md"
    json_path = REPORTS_DIR / "model_comparison.json"
    candidate_path = REPORTS_DIR / "candidate_manifest.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "| Modelo | Familia | AUC-PR | AUC-ROC | F1 | Precision | Recall | FP/h | Params | Edge | Optuna |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|\n")
        for row in rows:
            fp_h = row["test_fp_per_hour"]
            fp_h_text = "" if fp_h is None else f"{fp_h:.3f}"
            f.write(
                f"| {row['model_name']} "
                f"| {row['family']} "
                f"| {row['test_auc_pr']:.4f} "
                f"| {row['test_auc_roc']:.4f} "
                f"| {row['test_f1']:.4f} "
                f"| {row['test_precision']:.4f} "
                f"| {row['test_recall']:.4f} "
                f"| {fp_h_text} "
                f"| {row['parameter_count']} "
                f"| {row['edge_candidate']} "
                f"| {row['used_optuna']} |\n"
            )

    save_json(
        json_path,
        {
            "selection": selection_cfg,
            "results": results,
            "comparison_rows": rows,
            "best_overall": best_overall["model_name"],
            "best_edge": best_edge["model_name"] if best_edge else None,
        },
    )

    candidate = best_edge if best_edge is not None else best_overall
    candidate_row = _metric_row(candidate)
    save_json(
        candidate_path,
        {
            "selected_by": selection_cfg,
            "model_name": candidate["model_name"],
            "family": candidate["family"],
            "priority": candidate["priority"],
            "edge_candidate": candidate["edge_candidate"],
            "export_tflite": candidate["export_tflite"],
            "profile": candidate["profile"],
            "dataset": candidate["dataset"],
            "model_path": candidate["model_path"],
            "threshold": candidate["evaluation"]["threshold_from_val"],
            "summary_metrics": candidate_row,
            "metrics": candidate["evaluation"],
            "hpo": candidate["hpo"],
            "params": candidate["params"],
            "parameter_count": candidate.get("parameter_count"),
        },
    )

    log.info(f"CSV:       {csv_path}")
    log.info(f"Markdown:  {md_path}")
    log.info(f"Candidate: {candidate_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina todos os modelos habilitados.")
    parser.add_argument(
        "--models-cfg",
        type=Path,
        default=_default_models_cfg,
        help="Caminho para o YAML de modelos (default: config/model/models.yaml)",
    )
    args = parser.parse_args()
    models_config_path: Path = args.models_cfg

    ensure_directories()

    profile = PipelineProfile.from_yaml(PROFILE_PATH)

    models_cfg = load_yaml(models_config_path)
    models_to_train = {
        name: cfg
        for name, cfg in models_cfg.get("models", {}).items()
        if bool(cfg.get("enabled", False))
    }
    selection_cfg = models_cfg.get(
        "selection",
        {"metric": "auc_pr", "split": "test", "mode": "maximize"},
    )

    log.info("=" * 80)
    log.info(f"Config modelos: {models_config_path}")
    log.info(f"Modelos habilitados: {list(models_to_train)}")
    log.info("=" * 80)

    X_train, y_train = load_validated_split(DATASET_FILE, "train", profile)
    X_val, y_val = load_validated_split(DATASET_FILE, "val", profile)
    X_test, y_test = load_validated_split(DATASET_FILE, "test", profile)

    log.info(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    X_train_f = extract_statistical_features(X_train, profile.sampling_rate)
    X_val_f = extract_statistical_features(X_val, profile.sampling_rate)
    X_test_f = extract_statistical_features(X_test, profile.sampling_rate)

    log.info(f"Features: {X_train_f.shape[1]} dimensoes")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results: list[dict[str, Any]] = []

    for model_name, model_cfg in models_to_train.items():
        family = model_cfg.get("family")
        params = dict(model_cfg.get("params", {}))

        log.info("=" * 80)
        log.info(f"MODELO: {model_name}  |  FAMILIA: {family}")
        log.info("=" * 80)

        with mlflow.start_run(run_name=model_name):
            hpo_result = {
                "used_optuna": False,
                "best_params": params,
                "best_value": None,
                "n_trials": 0,
            }

            if family in ("classical_supervised", "classical_unsupervised"):
                hpo_result = tune_classical_model(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    X_train_f=X_train_f,
                    y_train=y_train,
                    X_val_f=X_val_f,
                    y_val=y_val,
                    profile=profile,
                    selection_metric=selection_cfg.get("metric", "auc_pr"),
                )
                final = train_classical_final(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    params=hpo_result["best_params"],
                    X_train_f=X_train_f,
                    y_train=y_train,
                    X_val_f=X_val_f,
                    y_val=y_val,
                    X_test_f=X_test_f,
                    y_test=y_test,
                    profile=profile,
                )
                result = {
                    "model_name": model_name,
                    "family": family,
                    "priority": model_cfg.get("priority", "baseline"),
                    "edge_candidate": bool(model_cfg.get("edge_candidate", False)),
                    "export_tflite": bool(model_cfg.get("export_tflite", False)),
                    "profile": profile.to_dict(),
                    "dataset": str(DATASET_FILE),
                    "model_path": final["model_path"],
                    "params": hpo_result["best_params"],
                    "hpo": hpo_result,
                    "feature_names": feature_names(),
                    "evaluation": final["evaluation"],
                }

            elif family == "neural_classifier":
                final = train_neural_classifier(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    profile=profile,
                )
                result = {
                    "model_name": model_name,
                    "family": family,
                    "priority": model_cfg.get("priority", "candidate"),
                    "edge_candidate": bool(model_cfg.get("edge_candidate", True)),
                    "export_tflite": bool(model_cfg.get("export_tflite", True)),
                    "profile": profile.to_dict(),
                    "dataset": str(DATASET_FILE),
                    "model_path": final["model_path"],
                    "params": params,
                    "hpo": hpo_result,
                    "evaluation": final["evaluation"],
                    "history": final["history"],
                    "parameter_count": final["parameter_count"],
                }

            elif family == "autoencoder":
                final = train_autoencoder(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    profile=profile,
                )
                result = {
                    "model_name": model_name,
                    "family": family,
                    "priority": model_cfg.get("priority", "candidate"),
                    "edge_candidate": bool(model_cfg.get("edge_candidate", True)),
                    "export_tflite": bool(model_cfg.get("export_tflite", True)),
                    "profile": profile.to_dict(),
                    "dataset": str(DATASET_FILE),
                    "model_path": final["model_path"],
                    "params": params,
                    "hpo": hpo_result,
                    "evaluation": final["evaluation"],
                    "history": final["history"],
                    "parameter_count": final["parameter_count"],
                }

            else:
                log.warning(f"Familia nao suportada: {family} â€” pulando {model_name}")
                continue

            log_result_to_mlflow(result)
            all_results.append(result)

            log.info(
                f"{model_name}: "
                f"test_auc_pr={result['evaluation']['test']['auc_pr']:.4f}  "
                f"test_f1={result['evaluation']['test']['f1']:.4f}"
            )

    if not all_results:
        log.error("Nenhum modelo foi treinado. Verifique models.yaml.")
        return

    best_overall = select_best_model(all_results, selection_cfg)
    best_edge = select_best_edge_candidate(all_results, selection_cfg)

    save_comparison_reports(
        results=all_results,
        best_overall=best_overall,
        best_edge=best_edge,
        selection_cfg=selection_cfg,
    )

    log.info("=" * 80)
    log.info("MELHOR MODELO GERAL")
    log.info("=" * 80)
    log.info(f"  {best_overall['model_name']}  AUC-PR={best_overall['evaluation']['test']['auc_pr']:.4f}")

    if best_edge:
        log.info("MELHOR EDGE CANDIDATE (para TFLite)")
        log.info(f"  {best_edge['model_name']}  AUC-PR={best_edge['evaluation']['test']['auc_pr']:.4f}")
    else:
        log.warning("Nenhum modelo marcado como edge_candidate=true.")


if __name__ == "__main__":
    main()

