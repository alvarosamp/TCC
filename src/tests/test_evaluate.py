from __future__ import annotations

import numpy as np
import pytest

from src.training.evaluate import (
    choose_threshold,
    evaluate_scores,
    evaluate_with_validation_threshold,
)

# Dados sinteticos simples: 10 amostras, 5 anomalias, scores perfeitos
@pytest.fixture
def perfect_data():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    return y, scores


@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(42)
    y = np.array([0] * 15 + [1] * 5, dtype=np.int32)
    scores = rng.uniform(0, 1, size=20).astype(np.float32)
    return y, scores


def test_choose_threshold_returns_valid_keys(perfect_data):
    y, scores = perfect_data
    result = choose_threshold(y, scores)
    assert "threshold" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_choose_threshold_perfect_data(perfect_data):
    y, scores = perfect_data
    result = choose_threshold(y, scores)
    assert result["f1"] == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= result["threshold"] <= 1.0


def test_evaluate_scores_keys(noisy_data):
    y, scores = noisy_data
    metrics = evaluate_scores(y, scores)
    for key in ("auc_pr", "auc_roc", "threshold", "f1", "precision", "recall", "confusion_matrix"):
        assert key in metrics


def test_evaluate_scores_values_are_finite(noisy_data):
    y, scores = noisy_data
    metrics = evaluate_scores(y, scores)
    assert np.isfinite(metrics["auc_pr"])
    assert np.isfinite(metrics["auc_roc"])
    assert np.isfinite(metrics["f1"])


def test_evaluate_scores_with_explicit_threshold(noisy_data):
    y, scores = noisy_data
    metrics = evaluate_scores(y, scores, threshold=0.5)
    assert metrics["threshold"] == pytest.approx(0.5)


def test_evaluate_scores_perfect_auc(perfect_data):
    y, scores = perfect_data
    metrics = evaluate_scores(y, scores)
    assert metrics["auc_pr"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["auc_roc"] == pytest.approx(1.0, abs=1e-4)


def test_confusion_matrix_shape(perfect_data):
    y, scores = perfect_data
    metrics = evaluate_scores(y, scores)
    cm = metrics["confusion_matrix"]
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)


def test_evaluate_with_validation_threshold_structure(noisy_data):
    y, scores = noisy_data
    result = evaluate_with_validation_threshold(
        y_val=y,
        scores_val=scores,
        y_test=y,
        scores_test=scores,
    )
    assert "val" in result
    assert "test" in result
    assert "threshold_from_val" in result
    assert "auc_pr" in result["test"]


def test_evaluate_with_validation_threshold_uses_val_threshold(perfect_data):
    y, scores = perfect_data
    result = evaluate_with_validation_threshold(
        y_val=y,
        scores_val=scores,
        y_test=y,
        scores_test=scores,
    )
    assert result["test"]["threshold"] == pytest.approx(result["threshold_from_val"])
