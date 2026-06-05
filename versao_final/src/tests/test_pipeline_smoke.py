from __future__ import annotations

import numpy as np

from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.features.statistical_features import extract_statistical_features
from src.training.evaluate import evaluate_with_validation_threshold
from src.training.model_registry import build_model


def test_classical_pipeline_smoke(
    tiny_profile_path,
    tiny_dataset_path,
):
    """
    Teste de comunicacao entre sistemas:

      profile
        -> schemas
        -> features
        -> model_registry
        -> evaluate

    Se esse teste passa, as partes principais conversam.
    """

    profile = PipelineProfile.from_yaml(tiny_profile_path)

    X_train, y_train = load_validated_split(
        tiny_dataset_path,
        "train",
        profile,
    )

    X_val, y_val = load_validated_split(
        tiny_dataset_path,
        "val",
        profile,
    )

    X_test, y_test = load_validated_split(
        tiny_dataset_path,
        "test",
        profile,
    )

    X_train_f = extract_statistical_features(
        X_train,
        sample_rate=profile.sampling_rate,
    )

    X_val_f = extract_statistical_features(
        X_val,
        sample_rate=profile.sampling_rate,
    )

    X_test_f = extract_statistical_features(
        X_test,
        sample_rate=profile.sampling_rate,
    )

    model = build_model(
        model_name="random_forest",
        params={
            "n_estimators": 5,
            "class_weight": "balanced",
        },
        seed=42,
    )

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
    assert "auc_pr" in metrics["test"]
    assert np.isfinite(metrics["test"]["auc_pr"])