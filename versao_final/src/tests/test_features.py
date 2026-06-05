from __future__ import annotations

import numpy as np

from src.features.statistical_features import (
    extract_statistical_features,
    feature_names,
)


def test_extract_statistical_features_shape():
    """
    Feature extractor deve transformar:
      (n, window_size) -> (n, n_features)
    """

    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 32)).astype(np.float32)

    features = extract_statistical_features(
        X,
        sample_rate=40.0,
    )

    assert features.shape[0] == 5
    assert features.shape[1] == len(feature_names())


def test_extract_statistical_features_no_nan():
    """
    Features nao devem conter NaN ou infinito.
    """

    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 32)).astype(np.float32)

    features = extract_statistical_features(
        X,
        sample_rate=40.0,
    )

    assert np.isfinite(features).all()


def test_extract_statistical_features_constant_signal_no_nan():
    """
    Mesmo sinal constante nao deve quebrar o pipeline.
    """

    X = np.ones((3, 32), dtype=np.float32)

    features = extract_statistical_features(
        X,
        sample_rate=40.0,
    )

    assert np.isfinite(features).all()