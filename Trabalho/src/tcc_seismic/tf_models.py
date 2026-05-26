from __future__ import annotations


def require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for neural models. Install with: pip install tensorflow"
        ) from exc
    return tf


def build_dense_ae(input_dim: int, latent_dim: int = 32, dropout: float = 0.1):
    tf = require_tensorflow()
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(latent)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="dense_ae")


def build_cnn1d_ae(input_dim: int):
    tf = require_tensorflow()
    inputs = tf.keras.layers.Input(shape=(input_dim, 1))
    x = tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    encoded = tf.keras.layers.MaxPooling1D(2, padding="same", name="encoded")(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(encoded)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Cropping1D((0, max(0, int(x.shape[1]) - input_dim)))(x)
    outputs = tf.keras.layers.Conv1D(1, 7, padding="same", activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="cnn1d_ae")


def build_lstm_ae(input_dim: int, timesteps: int = 50):
    tf = require_tensorflow()
    if input_dim % timesteps != 0:
        raise ValueError(f"input_dim={input_dim} must be divisible by timesteps={timesteps}")
    features = input_dim // timesteps
    inputs = tf.keras.layers.Input(shape=(timesteps, features))
    x = tf.keras.layers.LSTM(96, return_sequences=True)(inputs)
    latent = tf.keras.layers.LSTM(48, return_sequences=False, name="latent")(x)
    x = tf.keras.layers.RepeatVector(timesteps)(latent)
    x = tf.keras.layers.LSTM(48, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(96, return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(features))(x)
    return tf.keras.Model(inputs, outputs, name="lstm_ae")


def build_transformer_ae(
    input_dim: int,
    patch_size: int = 8,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout: float = 0.1,
):
    """Light Transformer autoencoder for 1D windows.

    The window is split into small temporal patches. Self-attention models
    relationships between distant parts of the signal, while the output is
    reconstructed back to the original window length.
    """

    tf = require_tensorflow()
    if input_dim % patch_size != 0:
        raise ValueError(f"input_dim={input_dim} must be divisible by patch_size={patch_size}")

    n_patches = input_dim // patch_size
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((n_patches, patch_size))(inputs)
    x = tf.keras.layers.Dense(d_model)(x)

    positions = tf.range(start=0, limit=n_patches, delta=1)
    pos_emb = tf.keras.layers.Embedding(input_dim=n_patches, output_dim=d_model)(positions)
    x = x + pos_emb

    for _ in range(num_blocks):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ]
        )(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = tf.keras.layers.Dense(patch_size)(x)
    outputs = tf.keras.layers.Reshape((input_dim,))(x)
    return tf.keras.Model(inputs, outputs, name="transformer_ae")

