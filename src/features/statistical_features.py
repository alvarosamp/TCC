from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

"""
Features genericas para janelas de series temporais.

Este modulo e propositalmente independente do dominio. Ele nao sabe se a janela
veio de sismologia, vibracao, audio ou corrente eletrica. Recebe apenas uma
matriz X com shape (n_janelas, n_amostras) e devolve uma matriz tabular de
features para modelos classicos.
"""


def _safe_stat(value: float) -> float:
    """Substitui NaN/inf por 0 para nao quebrar modelos classicos."""
    if not np.isfinite(value):
        return 0.0
    return float(value)


def time_domain_features(x: np.ndarray) -> list[float]:
    """Features no dominio do tempo."""
    x = np.asarray(x, dtype=np.float32)

    mean = _safe_stat(float(np.mean(x)))
    std = _safe_stat(float(np.std(x)))
    minimum = _safe_stat(float(np.min(x)))
    maximum = _safe_stat(float(np.max(x)))
    median = _safe_stat(float(np.median(x)))
    abs_mean = _safe_stat(float(np.mean(np.abs(x))))
    abs_peak = _safe_stat(float(np.max(np.abs(x))))
    rms = _safe_stat(float(np.sqrt(np.mean(x**2))))
    peak_to_peak = _safe_stat(float(maximum - minimum))
    energy = _safe_stat(float(np.sum(x**2)))
    skewness = _safe_stat(float(skew(x)))
    kurt = _safe_stat(float(kurtosis(x)))

    p05 = _safe_stat(float(np.percentile(x, 5)))
    p25 = _safe_stat(float(np.percentile(x, 25)))
    p75 = _safe_stat(float(np.percentile(x, 75)))
    p95 = _safe_stat(float(np.percentile(x, 95)))
    iqr = _safe_stat(float(p75 - p25))

    signs = np.signbit(x)
    zero_crossings = _safe_stat(float(np.count_nonzero(signs[1:] != signs[:-1])))
    zero_crossing_rate = _safe_stat(float(zero_crossings / max(len(x), 1)))

    crest_factor = _safe_stat(float(abs_peak / (rms + 1e-8)))

    return [
        mean,
        std,
        minimum,
        maximum,
        median,
        abs_mean,
        abs_peak,
        rms,
        crest_factor,
        peak_to_peak,
        energy,
        skewness,
        kurt,
        p05,
        p25,
        p75,
        p95,
        iqr,
        zero_crossings,
        zero_crossing_rate,
    ]


def _band_power(
    freqs: np.ndarray,
    power: np.ndarray,
    fmin: float,
    fmax: float,
    total_power: float,
) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return _safe_stat(float(np.sum(power[mask]) / (total_power + 1e-8)))


def _spectral_rolloff(freqs: np.ndarray, power: np.ndarray, ratio: float = 0.85) -> float:
    cumulative = np.cumsum(power)
    total = cumulative[-1] if len(cumulative) else 0.0
    if total <= 0:
        return 0.0
    idx = int(np.searchsorted(cumulative, ratio * total))
    idx = min(idx, len(freqs) - 1)
    return _safe_stat(float(freqs[idx]))


def frequency_domain_features(x: np.ndarray, sample_rate: float) -> list[float]:
    """Features espectrais usando FFT real."""
    x = np.asarray(x, dtype=np.float32)
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    power = magnitude**2
    total_power = float(np.sum(power)) + 1e-8

    dominant_idx = int(np.argmax(magnitude))
    dominant_freq = _safe_stat(float(freqs[dominant_idx]))
    spectral_centroid = _safe_stat(float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)))
    spectral_rolloff_85 = _spectral_rolloff(freqs, power, ratio=0.85)

    bandpower_0_3hz = _band_power(freqs, power, 0.0, 3.0, total_power)
    bandpower_0p5_3hz = _band_power(freqs, power, 0.5, 3.0, total_power)
    bandpower_3_8hz = _band_power(freqs, power, 3.0, 8.0, total_power)
    bandpower_8_15hz = _band_power(freqs, power, 8.0, 15.0, total_power)

    prob = power / total_power
    spectral_entropy = _safe_stat(float(-np.sum(prob * np.log(prob + 1e-8))))

    return [
        dominant_freq,
        spectral_centroid,
        spectral_rolloff_85,
        bandpower_0_3hz,
        bandpower_0p5_3hz,
        bandpower_3_8hz,
        bandpower_8_15hz,
        spectral_entropy,
    ]


def extract_features_from_window(x: np.ndarray, sample_rate: float) -> list[float]:
    features: list[float] = []
    features.extend(time_domain_features(x))
    features.extend(frequency_domain_features(x, sample_rate))
    return features


def extract_statistical_features(X: np.ndarray, sample_rate: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {X.shape}")
    rows = [extract_features_from_window(window, sample_rate) for window in X]
    return np.array(rows, dtype=np.float32)


def feature_names() -> list[str]:
    time_features = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "abs_mean",
        "abs_peak",
        "rms",
        "crest_factor",
        "peak_to_peak",
        "energy",
        "skewness",
        "kurtosis",
        "p05",
        "p25",
        "p75",
        "p95",
        "iqr",
        "zero_crossings",
        "zero_crossing_rate",
    ]
    freq_features = [
        "dominant_freq",
        "spectral_centroid",
        "spectral_rolloff_85",
        "bandpower_0_3hz",
        "bandpower_0p5_3hz",
        "bandpower_3_8hz",
        "bandpower_8_15hz",
        "spectral_entropy",
    ]
    return time_features + freq_features
