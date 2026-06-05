"""
Definicoes de arquiteturas neurais TinyML.

Exporta build_neural_model(model_name, window_size, params) -> keras.Model.

Modelos suportados:
  Classificadores  : tiny_cnn, tiny_tcn, lstm_classifier
  Autoencoders     : dense_autoencoder, cnn_autoencoder
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf


# ============================================================
# CLASSIFICADORES
# ============================================================

def build_tiny_cnn(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    """
    Conv1D empilhados com Global Average Pooling.

    Adequado para Edge — menos de 5 k parametros com config padrao.
    """
    filters = list(params.get("filters", [16, 32, 48]))
    kernels = list(params.get("kernels", [7, 5, 3]))
    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.15))
    lr = float(params.get("learning_rate", 0.001))

    inp = tf.keras.Input(shape=(window_size, 1))
    x = inp
    for f, k in zip(filters, kernels):
        x = tf.keras.layers.Conv1D(f, k, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")],
    )
    return model


def build_tiny_tcn(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    """
    Temporal Convolutional Network com dilatacao exponencial.

    Campo receptivo cobre toda a janela com poucos parametros.
    """
    filters = int(params.get("filters", 24))
    kernel_size = int(params.get("kernel_size", 5))
    dilations = list(params.get("dilations", [1, 2, 4, 8]))
    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.10))
    lr = float(params.get("learning_rate", 0.001))

    inp = tf.keras.Input(shape=(window_size, 1))
    x = inp
    for d in dilations:
        residual = x
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=d,
            activation="relu",
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        # projecao residual se necessario
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")],
    )
    return model


def build_lstm_classifier(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    units = int(params.get("units", 48))
    dense_units = int(params.get("dense_units", 24))
    dropout = float(params.get("dropout", 0.20))
    lr = float(params.get("learning_rate", 0.001))

    inp = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.LSTM(units, return_sequences=False)(inp)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")],
    )
    return model


# ============================================================
# AUTOENCODERS
# ============================================================

def build_dense_autoencoder(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    """
    Autoencoder denso.

    Entrada e saida: (n, window_size) — sem canal extra.
    Score de anomalia = erro de reconstrucao.
    """
    hidden_units = list(params.get("hidden_units", [256, 128]))
    latent_dim = int(params.get("latent_dim", 32))
    dropout = float(params.get("dropout", 0.10))
    lr = float(params.get("learning_rate", 0.001))

    inp = tf.keras.Input(shape=(window_size,))
    x = inp
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    encoded = tf.keras.layers.Dense(latent_dim, activation="relu")(x)

    x = encoded
    for units in reversed(hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    decoded = tf.keras.layers.Dense(window_size)(x)

    model = tf.keras.Model(inp, decoded)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
    )
    return model


def build_cnn_autoencoder(window_size: int, params: dict[str, Any]) -> tf.keras.Model:
    """
    Autoencoder convolucional.

    Entrada e saida: (n, window_size, 1).
    Score de anomalia = MSE medio por janela.
    """
    filters = list(params.get("filters", [32, 64, 128]))
    kernels = list(params.get("kernels", [7, 5, 3]))
    bottleneck = int(params.get("bottleneck_filters", 32))
    dropout = float(params.get("dropout", 0.10))
    lr = float(params.get("learning_rate", 0.001))

    inp = tf.keras.Input(shape=(window_size, 1))
    x = inp
    for f, k in zip(filters, kernels):
        x = tf.keras.layers.Conv1D(f, k, padding="same", activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(bottleneck, 1, padding="same", activation="relu")(x)
    for f, k in zip(reversed(filters), reversed(kernels)):
        x = tf.keras.layers.Conv1D(f, k, padding="same", activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Conv1D(1, 1, padding="same")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
    )
    return model


# ============================================================
# DISPATCHER
# ============================================================

_BUILDERS = {
    "tiny_cnn": build_tiny_cnn,
    "tiny_tcn": build_tiny_tcn,
    "lstm_classifier": build_lstm_classifier,
    "dense_autoencoder": build_dense_autoencoder,
    "cnn_autoencoder": build_cnn_autoencoder,
}


def build_neural_model(
    model_name: str,
    window_size: int,
    params: dict[str, Any],
) -> tf.keras.Model:
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Modelo neural '{model_name}' nao suportado. "
            f"Disponiveis: {list(_BUILDERS)}"
        )
    return _BUILDERS[model_name](window_size, params)
