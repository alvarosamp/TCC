from __future__ import annotations

import numpy as np


def extract_statistical_features(X: np.ndarray, sample_rate: float = 40.0) -> np.ndarray:
    """Feature set for classical ML baselines.

    The features are intentionally simple and explainable: time-domain summary,
    energy, zero-crossing behavior, and coarse frequency-domain descriptors.
    """

    X = np.asarray(X, dtype=np.float32)
    eps = 1e-12

    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_v = X.min(axis=1)
    max_v = X.max(axis=1)
    ptp = np.ptp(X, axis=1)
    rms = np.sqrt(np.mean(X * X, axis=1))
    energy = np.mean(X * X, axis=1)
    abs_mean = np.mean(np.abs(X), axis=1)
    crest = np.max(np.abs(X), axis=1) / (rms + eps)
    zcr = np.mean(np.diff(np.signbit(X), axis=1), axis=1)

    centered = X - mean[:, None]
    skew = np.mean(centered**3, axis=1) / (std**3 + eps)
    kurt = np.mean(centered**4, axis=1) / (std**4 + eps)

    spectrum = np.abs(np.fft.rfft(X, axis=1))
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / sample_rate)
    spec_sum = spectrum.sum(axis=1) + eps
    centroid = (spectrum * freqs[None, :]).sum(axis=1) / spec_sum
    dominant_freq = freqs[np.argmax(spectrum, axis=1)]

    def band_power(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            return np.zeros(len(X), dtype=np.float32)
        return np.mean(spectrum[:, mask] ** 2, axis=1)

    bands = [
        band_power(0.5, 2.0),
        band_power(2.0, 5.0),
        band_power(5.0, 10.0),
        band_power(10.0, 15.0),
    ]

    return np.column_stack(
        [
            mean,
            std,
            min_v,
            max_v,
            ptp,
            rms,
            energy,
            abs_mean,
            crest,
            zcr,
            skew,
            kurt,
            centroid,
            dominant_freq,
            *bands,
        ]
    ).astype(np.float32)

