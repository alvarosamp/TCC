from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
"""
Features genericas para janelas de series temporais.

Este modulo suporta:

1. Serie univariada:
   X.shape = (n_janelas, window_size)

2. Serie multivariada:
   X.shape = (n_janelas, window_size, n_canais)

Para modelos classicos, precisamos transformar cada janela em uma linha tabular.
Exemplo:

Entrada multivariada:
  uma janela com 128 amostras e 3 canais

Saida:
  uma lista de features:
    ch0_mean, ch0_std, ...
    ch1_mean, ch1_std, ...
    ch2_mean, ch2_std, ...
    corr_ch0_ch1, corr_ch0_ch2, corr_ch1_ch2
"""


def _safe_stat(value: float) -> float:
    """Substitui NaN/inf por 0 para nao quebrar modelos classicos."""
    if not np.isfinite(value):
        return 0.0
    return float(value)

def _as_3d_window(X: np.ndarray) -> np.ndarray:
    """
    Converte X para o formato padrao interno:
    (n_janelas, window_size, n_canais)
    
    Se vier univariado : (n_janelas, window_size)
    
    vira: (n_janelas, window_size, 1)
    
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X[:, :, np.newaxis]
    
    if X.ndim == 3:
        return X
    
    raise ValueError(
        f"Formato de X nao suportado: {X.shape}. "
        f"Use (n_janelas, window_size) ou "
        f"(n_janelas, window_size, n_canais)."
    )
    
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
    """Frequencia abaixo da qual esta acumulada uma porcentagem da energia.
    Com ration = 0.85, a rolloff_85 é a frequencia abaixo da qual esta acumulada 85% da energia.
    """
    cumulative = np.cumsum(power)

    if len(cumulative) == 0:
        return 0.0

    total = cumulative[-1]

    if total <= 0:
        return 0.0

    idx = int(np.searchsorted(cumulative, ratio * total))
    idx = min(idx, len(freqs) - 1)

    return _safe_stat(float(freqs[idx]))


def frequency_domain_features(x: np.ndarray, sample_rate: float) -> list[float]:
    """
    Extrai features no dominio da frequencia usando FFT real.

    Entrada:
      x.shape = (window_size,)

    sample_rate:
      frequencia de amostragem em Hz.
    """
    x = np.asarray(x, dtype=np.float32)

    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)

    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    power = magnitude**2
    total_power = float(np.sum(power)) + 1e-8

    dominant_idx = int(np.argmax(magnitude))

    dominant_freq = _safe_stat(float(freqs[dominant_idx]))
    spectral_centroid = _safe_stat(
        float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8))
    )
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


def _cross_channel_correlation(window_3d: np.ndarray) -> list[float]:
    """
    Correlação de Pearson entre todos os pares de canais de uma janela.

    window_3d.shape = (window_size, n_channels)
    Retorna n_channels*(n_channels-1)/2 valores.
    """
    n_channels = window_3d.shape[1]
    feats: list[float] = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            a = window_3d[:, i]
            b = window_3d[:, j]
            std_a = float(np.std(a)) + 1e-8
            std_b = float(np.std(b)) + 1e-8
            corr = float(np.mean((a - a.mean()) * (b - b.mean())) / (std_a * std_b))
            feats.append(_safe_stat(corr))
    return feats


def extract_features_from_window_multivariate(
    window: np.ndarray, sample_rate: float
) -> list[float]:
    """
    Extrai features de uma janela multivariada: (window_size, n_channels).

    Para cada canal: features temporais + espectrais.
    Depois: correlação cruzada entre pares de canais.
    """
    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 1:
        return extract_features_from_window(window, sample_rate)

    n_channels = window.shape[1]
    feats: list[float] = []
    for c in range(n_channels):
        feats.extend(extract_features_from_window(window[:, c], sample_rate))
    feats.extend(_cross_channel_correlation(window))
    return feats


def extract_statistical_features(X: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Extrai features de X.

    Suporta:
      univariado:   X.shape = (n_janelas, window_size)
      multivariado: X.shape = (n_janelas, window_size, n_channels)
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2:
        rows = [extract_features_from_window(window, sample_rate) for window in X]
    elif X.ndim == 3:
        rows = [
            extract_features_from_window_multivariate(X[i], sample_rate)
            for i in range(len(X))
        ]
    else:
        raise ValueError(f"Input must be 2D or 3D array, got shape {X.shape}")

    return np.array(rows, dtype=np.float32)


def feature_names(n_channels: int = 1) -> list[str]:
    """Retorna nomes das features. Para multivariado, prefixados por canal."""
    if n_channels > 1:
        names: list[str] = []
        base = _base_feature_names()
        for c in range(n_channels):
            names.extend([f"ch{c}_{f}" for f in base])
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                names.append(f"corr_ch{i}_ch{j}")
        return names
    return _base_feature_names()


def _base_feature_names() -> list[str]:
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


# manter compatibilidade com chamadas antigas sem argumento
def feature_names_univariate() -> list[str]:
    return _base_feature_names()
