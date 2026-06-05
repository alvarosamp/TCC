"""
Smoke test do pipeline de treino completo.

Cobre:
  - Pipeline classico: features -> Optuna (3 trials) -> treino -> avaliacao
  - Pipeline neural:  tiny_cnn com 10 epochs (marcado slow, requer TF)

Nao depende de neural_models.py nem de arquivos de dados reais.
Usa as fixtures tiny_profile_path e tiny_dataset_path do conftest.
"""
from __future__ import annotations

import numpy as np
import pytest
import optuna

from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.features.statistical_features import extract_statistical_features
from src.training.evaluate import evaluate_scores, evaluate_with_validation_threshold
from src.training.hpo import suggest_params
from src.training.model_registry import build_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

_LR_SEARCH_SPACE = {
    "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
}
_LR_BASE_PARAMS = {"class_weight": "balanced", "max_iter": 200, "C": 1.0}


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def _load_splits(tiny_dataset_path, profile):
    X_train, y_train = load_validated_split(tiny_dataset_path, "train", profile)
    X_val, y_val = load_validated_split(tiny_dataset_path, "val", profile)
    X_test, y_test = load_validated_split(tiny_dataset_path, "test", profile)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _extract_features(X_train, X_val, X_test, sampling_rate):
    X_train_f = extract_statistical_features(X_train, sampling_rate)
    X_val_f = extract_statistical_features(X_val, sampling_rate)
    X_test_f = extract_statistical_features(X_test, sampling_rate)
    return X_train_f, X_val_f, X_test_f


def _optuna_tune(model_name, base_params, search_space, n_trials,
                 X_train_f, y_train, X_val_f, y_val):
    def objective(trial):
        params = dict(base_params)
        params.update(suggest_params(trial, search_space))
        m = build_model(model_name, params, SEED)
        m.fit(X_train_f, y_train)
        scores = m.predict_proba(X_val_f)[:, 1]
        return float(evaluate_scores(y_val, scores, threshold=None)["auc_pr"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = dict(base_params)
    best.update(study.best_params)
    return best, study.best_value


# ----------------------------------------------------------------
# testes classicos
# ----------------------------------------------------------------

def test_classical_smoke_optuna(tiny_profile_path, tiny_dataset_path):
    """Logistic Regression com 3 trials Optuna — deve rodar em segundos."""
    profile = PipelineProfile.from_yaml(tiny_profile_path)

    X_train, y_train, X_val, y_val, X_test, y_test = _load_splits(
        tiny_dataset_path, profile
    )
    X_train_f, X_val_f, X_test_f = _extract_features(
        X_train, X_val, X_test, profile.sampling_rate
    )

    best_params, best_val = _optuna_tune(
        model_name="logistic_regression",
        base_params=_LR_BASE_PARAMS,
        search_space=_LR_SEARCH_SPACE,
        n_trials=3,
        X_train_f=X_train_f, y_train=y_train,
        X_val_f=X_val_f, y_val=y_val,
    )

    model = build_model("logistic_regression", best_params, SEED)
    model.fit(X_train_f, y_train)
    scores_val = model.predict_proba(X_val_f)[:, 1]
    scores_test = model.predict_proba(X_test_f)[:, 1]

    metrics = evaluate_with_validation_threshold(
        y_val=y_val,
        scores_val=scores_val,
        y_test=y_test,
        scores_test=scores_test,
    )

    assert "val" in metrics
    assert "test" in metrics
    assert np.isfinite(metrics["test"]["auc_pr"])
    assert 0.0 <= metrics["test"]["auc_pr"] <= 1.0
    assert np.isfinite(metrics["test"]["f1"])


# ----------------------------------------------------------------
# teste neural
# ----------------------------------------------------------------

@pytest.mark.slow
def test_neural_classifier_smoke_10epochs(tiny_profile_path, tiny_dataset_path):
    """
    Tiny CNN com 10 epochs sobre dataset sintetico.

    Arquitetura minima: 2x Conv1D(8), GAP, Dense(8) -> sigmoid.
    Serve para validar que o fluxo de dados ate evaluate funciona.

    Marcar com: pytest -m slow
    """
    import tensorflow as tf

    profile = PipelineProfile.from_yaml(tiny_profile_path)

    X_train, y_train, X_val, y_val, X_test, y_test = _load_splits(
        tiny_dataset_path, profile
    )

    def as_conv(X: np.ndarray) -> np.ndarray:
        return X[..., np.newaxis].astype(np.float32)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    inp = tf.keras.Input(shape=(profile.window_size, 1))
    x = tf.keras.layers.Conv1D(8, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")],
    )

    labels, counts = np.unique(y_train.astype(int), return_counts=True)
    total = float(len(y_train))
    class_weight = {
        int(lbl): total / (float(len(labels)) * float(cnt))
        for lbl, cnt in zip(labels, counts)
    }

    model.fit(
        as_conv(X_train),
        y_train,
        validation_data=(as_conv(X_val), y_val),
        epochs=10,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc_pr",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    scores_val = model.predict(as_conv(X_val), verbose=0).reshape(-1)
    scores_test = model.predict(as_conv(X_test), verbose=0).reshape(-1)

    metrics = evaluate_with_validation_threshold(
        y_val=y_val,
        scores_val=scores_val,
        y_test=y_test,
        scores_test=scores_test,
    )

    assert "val" in metrics
    assert "test" in metrics
    assert np.isfinite(metrics["test"]["auc_pr"])
    assert 0.0 <= metrics["test"]["auc_pr"] <= 1.0
