from __future__ import annotations

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


# ============================================================
# OBJETIVO DO ARQUIVO
# ============================================================
#
# Treina todos os modelos habilitados em configs/models.yaml.
#
# Suporta familias:
#
#   classical_supervised:
#     - logistic_regression
#     - random_forest
#     - extra_trees
#
#   classical_unsupervised:
#     - isolation_forest
#
#   neural_classifier:
#     - tiny_cnn
#     - tiny_tcn
#     - lstm_classifier
#     - transformer_tiny
#
#   autoencoder:
#     - dense_autoencoder
#     - cnn_autoencoder
#
# Fluxo:
#   1. carrega dataset validado
#   2. le configs/models.yaml
#   3. treina enabled=true
#   4. roda Optuna nos modelos classicos quando habilitado
#   5. avalia em validacao e teste
#   6. registra no MLflow
#   7. gera tabela comparativa
#   8. salva candidate_manifest.json com melhor modelo


log = logging.getLogger("train_all")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


MODELS_CONFIG_PATH = ROOT_DIR / "configs" / "models.yaml"


# ============================================================
# IO
# ============================================================

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Le arquivo YAML."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Salva JSON legivel."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def enabled_models(models_cfg: dict[str, Any]) -> dict[str, Any]:
    """Retorna apenas modelos com enabled=true."""

    return {
        name: cfg
        for name, cfg in models_cfg.get("models", {}).items()
        if bool(cfg.get("enabled", False))
    }


# ============================================================
# PREPARACAO PARA KERAS
# ============================================================

def as_conv_input(X: np.ndarray) -> np.ndarray:
    """
    Converte X para entrada Conv1D/LSTM/Transformer.

    De:
      (n, window_size)

    Para:
      (n, window_size, 1)
    """

    return X[..., np.newaxis].astype(np.float32, copy=False)


def class_weight_from_labels(y: np.ndarray) -> dict[int, float]:
    """
    Calcula pesos por classe.

    Ajuda em datasets desbalanceados.
    """

    labels, counts = np.unique(
        y.astype(int),
        return_counts=True,
    )

    total = float(len(y))
    n_classes = float(len(labels))

    return {
        int(label): total / (n_classes * float(count))
        for label, count in zip(labels, counts)
    }


# ============================================================
# SCORES CLASSICOS
# ============================================================

def supervised_scores(model, X_features: np.ndarray) -> np.ndarray:
    """
    Score continuo para modelo supervisionado.

    Maior score = maior chance de anomalia.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_features)[:, 1].astype(np.float32)

    if hasattr(model, "decision_function"):
        return np.asarray(
            model.decision_function(X_features),
            dtype=np.float32,
        )

    raise TypeError(
        f"Modelo {type(model).__name__} nao possui score supervisionado."
    )


def unsupervised_scores(model, X_features: np.ndarray) -> np.ndarray:
    """
    Score continuo para modelo nao supervisionado.

    IsolationForest retorna score maior para normal.
    Por isso usamos sinal negativo.
    """

    if hasattr(model, "score_samples"):
        return (-model.score_samples(X_features)).astype(np.float32)

    if hasattr(model, "decision_function"):
        return (-model.decision_function(X_features)).astype(np.float32)

    raise TypeError(
        f"Modelo {type(model).__name__} nao possui score nao supervisionado."
    )


# ============================================================
# OPTUNA PARA CLASSICOS
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
    """
    Treina uma vez e avalia na validacao.

    Usado pelo Optuna.
    """

    model = build_model(
        model_name=model_name,
        params=params,
        seed=SEED,
    )

    if family == "classical_supervised":
        model.fit(X_train_f, y_train)
        scores_val = supervised_scores(model, X_val_f)

    elif family == "classical_unsupervised":
        normal_mask = y_train == profile.normal_label
        model.fit(X_train_f[normal_mask])
        scores_val = unsupervised_scores(model, X_val_f)

    else:
        raise ValueError(f"Familia classica nao suportada: {family}")

    val_metrics = evaluate_scores(
        y_true=y_val,
        scores=scores_val,
        threshold=None,
    )

    return {
        "model": model,
        "scores_val": scores_val,
        "val_metrics": val_metrics,
    }


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
    """
    Roda Optuna para modelos classicos.

    Para redes neurais, deixamos Optuna desligado por enquanto,
    porque e muito mais caro no Colab.
    """

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

        trial_params.update(
            suggest_params(
                trial=trial,
                search_space=search_space,
            )
        )

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


# ============================================================
# TREINO CLASSICO FINAL
# ============================================================

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
    """
    Treina modelo classico final e avalia em validacao/teste.
    """

    family = model_cfg["family"]

    model = build_model(
        model_name=model_name,
        params=params,
        seed=SEED,
    )

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

    return {
        "model_path": str(model_path),
        "evaluation": evaluation,
    }


# ============================================================
# TREINO NEURAL CLASSIFIER
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
    """
    Treina rede neural classificadora.

    Exemplos:
      tiny_cnn
      tiny_tcn
      lstm_classifier
      transformer_tiny
    """

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
        as_conv_input(X_val),
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    scores_test = model.predict(
        as_conv_input(X_test),
        batch_size=batch_size,
        verbose=0,
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
# TREINO AUTOENCODER
# ============================================================

def reconstruction_scores(
    model: tf.keras.Model,
    X: np.ndarray,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Calcula erro de reconstrucao por janela.

    Quanto maior o erro, maior a chance de anomalia.
    """

    if model_name == "dense_autoencoder":
        X_input = X.astype(np.float32, copy=False)
        reconstructed = model.predict(
            X_input,
            batch_size=batch_size,
            verbose=0,
        )

        errors = np.mean(
            (X_input - reconstructed) ** 2,
            axis=1,
        )

    else:
        X_input = as_conv_input(X)
        reconstructed = model.predict(
            X_input,
            batch_size=batch_size,
            verbose=0,
        )

        errors = np.mean(
            (X_input - reconstructed) ** 2,
            axis=(1, 2),
        )

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
    """
    Treina autoencoder somente com dados normais.

    Avaliacao:
      score = erro de reconstrucao
    """

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

    scores_val = reconstruction_scores(
        model=model,
        X=X_val,
        model_name=model_name,
        batch_size=batch_size,
    )

    scores_test = reconstruction_scores(
        model=model,
        X=X_test,
        model_name=model_name,
        batch_size=batch_size,
    )

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
# LOGGING E RELATORIOS
# ============================================================

def log_result_to_mlflow(result: dict[str, Any]) -> None:
    """
    Registra resultado no MLflow.
    """

    model_name = result["model_name"]
    evaluation = result["evaluation"]

    metrics_path = REPORTS_DIR / f"{model_name}_metrics.json"
    save_json(metrics_path, result)

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("family", result["family"])
    mlflow.log_param("profile", result["profile"]["profile_name"] + ":" + result["profile"]["profile_version"])
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


def metric_value(
    result: dict[str, Any],
    split: str,
    metric: str,
) -> float:
    """Pega valor de metrica de um resultado."""

    return float(result["evaluation"][split][metric])


def select_best_model(
    results: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Seleciona o melhor modelo conforme configs/models.yaml.

    Exemplo:
      selection:
        metric: auc_pr
        split: test
        mode: maximize
    """

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


def save_comparison_reports(
    results: list[dict[str, Any]],
    best: dict[str, Any],
    selection_cfg: dict[str, Any],
) -> None:
    """
    Salva CSV, Markdown, JSON e candidate_manifest.
    """

    rows = []

    for result in results:
        rows.append(
            {
                "model_name": result["model_name"],
                "family": result["family"],
                "priority": result["priority"],
                "edge_candidate": result["edge_candidate"],
                "export_tflite": result["export_tflite"],
                "used_optuna": result["hpo"]["used_optuna"],
                "parameter_count": result.get("parameter_count", ""),
                "val_auc_pr": result["evaluation"]["val"]["auc_pr"],
                "val_f1": result["evaluation"]["val"]["f1"],
                "test_auc_pr": result["evaluation"]["test"]["auc_pr"],
                "test_f1": result["evaluation"]["test"]["f1"],
                "threshold_from_val": result["evaluation"]["threshold_from_val"],
                "model_path": result["model_path"],
            }
        )

    csv_path = REPORTS_DIR / "model_comparison.csv"
    md_path = REPORTS_DIR / "model_comparison.md"
    json_path = REPORTS_DIR / "model_comparison.json"
    candidate_path = REPORTS_DIR / "candidate_manifest.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "| Modelo | Familia | AUC-PR Test | F1 Test | Params | Edge | Optuna |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|\n")

        for row in rows:
            f.write(
                f"| {row['model_name']} "
                f"| {row['family']} "
                f"| {row['test_auc_pr']:.4f} "
                f"| {row['test_f1']:.4f} "
                f"| {row['parameter_count']} "
                f"| {row['edge_candidate']} "
                f"| {row['used_optuna']} |\n"
            )

    save_json(
        json_path,
        {
            "selection": selection_cfg,
            "results": results,
            "best_model": best["model_name"],
        },
    )

    save_json(
        candidate_path,
        {
            "selected_by": selection_cfg,
            "model_name": best["model_name"],
            "family": best["family"],
            "priority": best["priority"],
            "edge_candidate": best["edge_candidate"],
            "export_tflite": best["export_tflite"],
            "profile": best["profile"],
            "dataset": best["dataset"],
            "model_path": best["model_path"],
            "threshold": best["evaluation"]["threshold_from_val"],
            "metrics": best["evaluation"],
            "hpo": best["hpo"],
            "params": best["params"],
            "parameter_count": best.get("parameter_count"),
        },
    )

    log.info(f"Comparacao CSV: {csv_path}")
    log.info(f"Comparacao MD:  {md_path}")
    log.info(f"Candidate:      {candidate_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Comando:
      python -m src.training.train_all
    """

    ensure_directories()

    profile = PipelineProfile.from_yaml(PROFILE_PATH)

    models_cfg = load_yaml(MODELS_CONFIG_PATH)
    models_to_train = {
        name: cfg
        for name, cfg in models_cfg.get("models", {}).items()
        if bool(cfg.get("enabled", False))
    }

    selection_cfg = models_cfg.get(
        "selection",
        {
            "metric": "auc_pr",
            "split": "test",
            "mode": "maximize",
        },
    )

    log.info("=" * 80)
    log.info("CARREGANDO DATASET")
    log.info("=" * 80)

    X_train, y_train = load_validated_split(DATASET_FILE, "train", profile)
    X_val, y_val = load_validated_split(DATASET_FILE, "val", profile)
    X_test, y_test = load_validated_split(DATASET_FILE, "test", profile)

    log.info(f"X_train: {X_train.shape}")
    log.info(f"X_val:   {X_val.shape}")
    log.info(f"X_test:  {X_test.shape}")

    log.info("=" * 80)
    log.info("EXTRAINDO FEATURES CLASSICAS")
    log.info("=" * 80)

    X_train_f = extract_statistical_features(
        X_train,
        profile.sampling_rate,
    )

    X_val_f = extract_statistical_features(
        X_val,
        profile.sampling_rate,
    )

    X_test_f = extract_statistical_features(
        X_test,
        profile.sampling_rate,
    )

    log.info(f"X_train_f: {X_train_f.shape}")
    log.info(f"X_val_f:   {X_val_f.shape}")
    log.info(f"X_test_f:  {X_test_f.shape}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results = []

    for model_name, model_cfg in models_to_train.items():
        family = model_cfg.get("family")
        params = dict(model_cfg.get("params", {}))

        log.info("=" * 80)
        log.info(f"MODELO: {model_name} | FAMILY: {family}")
        log.info("=" * 80)

        with mlflow.start_run(run_name=model_name):
            hpo_result = {
                "used_optuna": False,
                "best_params": params,
                "best_value": None,
                "n_trials": 0,
            }

            if family in [
                "classical_supervised",
                "classical_unsupervised",
            ]:
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
                raise ValueError(f"Familia nao suportada: {family}")

            log_result_to_mlflow(result)
            all_results.append(result)

            log.info(
                f"{model_name}: "
                f"test_auc_pr={result['evaluation']['test']['auc_pr']:.4f} "
                f"test_f1={result['evaluation']['test']['f1']:.4f}"
            )

    best = select_best_model(
        results=all_results,
        selection_cfg=selection_cfg,
    )

    save_comparison_reports(
        results=all_results,
        best=best,
        selection_cfg=selection_cfg,
    )

    log.info("=" * 80)
    log.info("MELHOR MODELO SELECIONADO")
    log.info("=" * 80)
    log.info(f"Modelo: {best['model_name']}")
    log.info(f"Familia: {best['family']}")
    log.info(f"AUC-PR test: {best['evaluation']['test']['auc_pr']:.4f}")
    log.info(f"F1 test:     {best['evaluation']['test']['f1']:.4f}")


if __name__ == "__main__":
    main()