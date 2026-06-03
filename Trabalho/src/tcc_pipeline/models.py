from __future__ import annotations

from typing import Any


def require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for neural training. "
            "Install with: pip install -r requirements-models.txt"
        ) from exc
    return tf


def build_tiny_cnn_classifier(
    window_size: int,
    learning_rate: float = 1e-3,
    dropout: float = 0.15,
) -> Any:
    """Compact 1D CNN classifier for edge-oriented time-series windows."""

    tf = require_tensorflow()
    inputs = tf.keras.layers.Input(shape=(window_size, 1), name="window")
    x = tf.keras.layers.Conv1D(16, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)

    model = tf.keras.Model(inputs, outputs, name="tiny_cnn_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
        ],
    )
    return model


def _tcn_block(tf: Any, x: Any, filters: int, kernel_size: int, dilation: int, dropout: float) -> Any:
    residual = x
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation,
        activation="relu",
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation,
        activation="relu",
    )(x)
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
    return tf.keras.layers.Add()([x, residual])


def build_tiny_tcn_classifier(
    window_size: int,
    learning_rate: float = 1e-3,
    filters: int = 24,
    kernel_size: int = 5,
    dropout: float = 0.10,
) -> Any:
    """Small TCN classifier suitable for later TensorFlow Lite Micro export."""

    tf = require_tensorflow()
    inputs = tf.keras.layers.Input(shape=(window_size, 1), name="window")
    x = inputs
    for dilation in [1, 2, 4, 8]:
        x = _tcn_block(tf, x, filters, kernel_size, dilation, dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)

    model = tf.keras.Model(inputs, outputs, name="tiny_tcn_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
        ],
    )
    return model


def build_classifier(model_name: str, window_size: int, **kwargs: Any) -> Any:
    builders = {
        "tiny_cnn": build_tiny_cnn_classifier,
        "tiny_tcn": build_tiny_tcn_classifier,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Options: {sorted(builders)}")
    return builders[model_name](window_size=window_size, **kwargs)
