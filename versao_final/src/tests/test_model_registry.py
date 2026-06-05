from __future__ import annotations

import numpy as np

from src.training.model_registry import build_model


def test_build_random_forest_and_fit():
    """
    Registry deve construir Random Forest treinavel.
    """

    X = np.random.default_rng(42).normal(size=(20, 5))
    y = np.array([0] * 10 + [1] * 10)

    model = build_model(
        model_name="random_forest",
        params={
            "n_estimators": 5,
            "class_weight": "balanced",
        },
        seed=42,
    )

    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (20,)


def test_build_isolation_forest_and_score():
    """
    Registry deve construir Isolation Forest com score_samples.
    """

    X = np.random.default_rng(42).normal(size=(20, 5))

    model = build_model(
        model_name="isolation_forest",
        params={
            "n_estimators": 5,
            "contamination": 0.1,
        },
        seed=42,
    )

    model.fit(X)

    scores = model.score_samples(X)

    assert scores.shape == (20,)