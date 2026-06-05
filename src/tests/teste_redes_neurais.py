from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.training.neural_models import build_neural_model


def test_build_tiny_cnn_forward_pass():
    """
    Tiny CNN deve aceitar entrada (n, window_size, 1).
    """

    tf.keras.backend.clear_session()

    model = build_neural_model(
        model_name="tiny_cnn",
        window_size=32,
        params={
            "filters": [4, 8],
            "kernels": [3, 3],
            "dense_units": 4,
            "dropout": 0.0,
            "learning_rate": 0.001,
        },
    )

    X = np.random.default_rng(42).normal(
        size=(2, 32, 1),
    ).astype(np.float32)

    y = model.predict(X, verbose=0)

    assert y.shape == (2, 1)


def test_build_tiny_tcn_forward_pass():
    """
    Tiny TCN deve aceitar entrada (n, window_size, 1).
    """

    tf.keras.backend.clear_session()

    model = build_neural_model(
        model_name="tiny_tcn",
        window_size=32,
        params={
            "filters": 4,
            "kernel_size": 3,
            "dilations": [1, 2],
            "dense_units": 4,
            "dropout": 0.0,
            "learning_rate": 0.001,
        },
    )

    X = np.random.default_rng(42).normal(
        size=(2, 32, 1),
    ).astype(np.float32)

    y = model.predict(X, verbose=0)

    assert y.shape == (2, 1)


def test_build_dense_autoencoder_forward_pass():
    """
    Dense AE deve reconstruir shape (n, window_size).
    """

    tf.keras.backend.clear_session()

    model = build_neural_model(
        model_name="dense_autoencoder",
        window_size=32,
        params={
            "hidden_units": [16, 8],
            "latent_dim": 4,
            "dropout": 0.0,
            "learning_rate": 0.001,
        },
    )

    X = np.random.default_rng(42).normal(
        size=(2, 32),
    ).astype(np.float32)

    y = model.predict(X, verbose=0)

    assert y.shape == (2, 32)


def test_build_cnn_autoencoder_forward_pass():
    """
    CNN AE deve reconstruir shape (n, window_size, 1).
    """

    tf.keras.backend.clear_session()

    model = build_neural_model(
        model_name="cnn_autoencoder",
        window_size=32,
        params={
            "filters": [4, 8],
            "kernels": [3, 3],
            "bottleneck_filters": 4,
            "learning_rate": 0.001,
        },
    )

    X = np.random.default_rng(42).normal(
        size=(2, 32, 1),
    ).astype(np.float32)

    y = model.predict(X, verbose=0)

    assert y.shape == (2, 32, 1)