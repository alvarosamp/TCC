"""
Arquiteturas neurais leves para TinyML.

O arquivo aceita uma configuracao simples, mas tambem suporta parametros usados
nas rodadas de Optuna, como separable conv, batch norm e pooling avgmax. Assim o
pipeline do repositorio consegue representar melhor os experimentos reais.
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf


def _conv1d_layer(filters: int, kernel_size: int, conv_type: str, **kwargs: Any):
    if conv_type == "separable":
        regularizer = kwargs.pop("kernel_regularizer", None)
        if regularizer is not None:
            kwargs.setdefault("depthwise_regularizer", regularizer)
            kwargs.setdefault("pointwise_regularizer", regularizer)
        return tf.keras.layers.SeparableConv1D(filters, kernel_size, **kwargs)
    return tf.keras.layers.Conv1D(filters, kernel_size, **kwargs)


def _activation_block(x, use_batch_norm: bool):
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def _head_pooling(x, mode: str):
    if mode == "avgmax":
        avg = tf.keras.layers.GlobalAveragePooling1D()(x)
        mx = tf.keras.layers.GlobalMaxPooling1D()(x)
        return tf.keras.layers.Concatenate()([avg, mx])
    if mode == "max":
        return tf.keras.layers.GlobalMaxPooling1D()(x)
    return tf.keras.layers.GlobalAveragePooling1D()(x)


def _compile_classifier(model: tf.keras.Model, params: dict[str, Any]) -> tf.keras.Model:
    lr = float(params.get("learning_rate", params.get("lr", 0.001)))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")],
    )
    return model


def build_tiny_cnn(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    filters = list(params.get("filters", []))
    if not filters:
        n_blocks = int(params.get("n_blocks", 3))
        base_filters = int(params.get("base_filters", 16))
        filters = [base_filters * (i + 1) for i in range(n_blocks)]

    kernels = list(params.get("kernels", []))
    if not kernels:
        kernel_first = int(params.get("kernel_first", 7))
        kernel_other = int(params.get("kernel_other", 5))
        kernels = [kernel_first] + [kernel_other] * (len(filters) - 1)

    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.15))
    spatial_dropout = float(params.get("spatial_dropout", 0.0))
    conv_type = str(params.get("conv_type", "conv"))
    use_batch_norm = bool(params.get("use_batch_norm", False))
    head_pooling = str(params.get("head_pooling", params.get("pooling", "avg")))
    l2_reg = float(params.get("l2_reg", 0.0))
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    inp = tf.keras.Input(shape=(window_size, 1))
    x = inp
    for f, k in zip(filters, kernels):
        x = _conv1d_layer(
            int(f),
            int(k),
            conv_type,
            padding="same",
            use_bias=not use_batch_norm,
            kernel_regularizer=regularizer,
        )(x)
        x = _activation_block(x, use_batch_norm)
        if spatial_dropout > 0:
            x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    x = _head_pooling(x, head_pooling)
    if dense_units > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizer)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile_classifier(tf.keras.Model(inp, out), params)


def build_tiny_tcn(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    filters = int(params.get("filters", 24))
    kernel_size = int(params.get("kernel_size", 5))
    n_blocks = int(params.get("n_blocks", 0))
    dilation_base = int(params.get("dilation_base", 2))
    dilations = list(params.get("dilations", []))
    if not dilations:
        if n_blocks <= 0:
            n_blocks = 4
        dilations = [dilation_base**i for i in range(n_blocks)]

    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.10))
    spatial_dropout = float(params.get("spatial_dropout", 0.0))
    padding = str(params.get("padding", "causal"))
    conv_type = str(params.get("conv_type", "conv"))
    use_batch_norm = bool(params.get("use_batch_norm", False))
    head_pooling = str(params.get("head_pooling", "avg"))
    l2_reg = float(params.get("l2_reg", 0.0))
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    inp = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="same")(inp)

    for d in dilations:
        residual = x
        x = _conv1d_layer(
            filters,
            kernel_size,
            conv_type,
            padding=padding,
            dilation_rate=int(d),
            use_bias=not use_batch_norm,
            kernel_regularizer=regularizer,
        )(x)
        x = _activation_block(x, use_batch_norm)
        if spatial_dropout > 0:
            x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(
            filters, kernel_size=1, padding="same",
            use_bias=not use_batch_norm, kernel_regularizer=regularizer,
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(
                filters, 1, padding="same", kernel_regularizer=regularizer,
            )(residual)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation("relu")(x)

    x = _head_pooling(x, head_pooling)
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizer)(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile_classifier(tf.keras.Model(inp, out), params)


def build_lstm_classifier(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    units = int(params.get("units", 48))
    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.20))

    inp = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.LSTM(units, return_sequences=False)(inp)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile_classifier(tf.keras.Model(inp, out), params)


def build_dense_autoencoder(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    hidden_units = list(params.get("hidden_units", [256, 128]))
    latent_dim = int(params.get("latent_dim", 32))
    dropout = float(params.get("dropout", 0.10))
    lr = float(params.get("learning_rate", params.get("lr", 0.001)))

    inp = tf.keras.Input(shape=(window_size,))
    x = inp
    for units in hidden_units:
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    encoded = tf.keras.layers.Dense(latent_dim, activation="relu")(x)

    x = encoded
    for units in reversed(hidden_units):
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    decoded = tf.keras.layers.Dense(window_size)(x)

    model = tf.keras.Model(inp, decoded)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def build_cnn_autoencoder(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    filters = list(params.get("filters", [32, 64, 128]))
    kernels = list(params.get("kernels", [7, 5, 3]))
    bottleneck = int(params.get("bottleneck_filters", 32))
    dropout = float(params.get("dropout", 0.10))
    lr = float(params.get("learning_rate", params.get("lr", 0.001)))

    inp = tf.keras.Input(shape=(window_size, 1))
    x = inp
    for f, k in zip(filters, kernels):
        x = tf.keras.layers.Conv1D(int(f), int(k), padding="same", activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(bottleneck, 1, padding="same", activation="relu")(x)
    for f, k in zip(reversed(filters), reversed(kernels)):
        x = tf.keras.layers.Conv1D(int(f), int(k), padding="same", activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Conv1D(1, 1, padding="same")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


_BUILDERS = {
    "tiny_cnn": build_tiny_cnn,
    "tiny_tcn": build_tiny_tcn,
    "lstm_classifier": build_lstm_classifier,
    "dense_autoencoder": build_dense_autoencoder,
    "cnn_autoencoder": build_cnn_autoencoder,
}


def build_neural_model(model_name: str, window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    if model_name not in _BUILDERS:
        raise ValueError(f"Modelo neural '{model_name}' nao suportado. Disponiveis: {list(_BUILDERS)}")
    return _BUILDERS[model_name](window_size, params)

