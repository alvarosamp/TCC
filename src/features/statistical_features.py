from __future__ import annotations
from curses import window
import numpy as np
from scipy.stats import kurtosis, skew
'''
Extrai features genericas de janelas temporais
'''

def _safe_stat(value: float) -> float:
    '''
    Garantee que o valor é finito, substituindo inf e nan por 0
    '''
    if np.insnan(value) or np.isinf(value):
        return 0.0
    return float(value)

def time_domain_features(X:np.ndarray) -> list[float]:
    X = np.asarray(X, dtype = np.float32)
    mean = _safe_stat(float(np.mean(X)))
    std = _safe_stat(float(np.std(X)))
    minimum = _safe_stat(float(np.min(X)))
    maximum = _safe_stat(float(np.max(X)))
    median = _safe_stat(float(np.median(X)))
    abs_mean = _safe_stat(float(np.mean(np.abs(X))))
    rms = _safe_stat(float(np.sqrt(np.mean(X**2))))
    peak_to_peak = _safe_stat(float(maximum - minimum))
    energy = _safe_stat(float(np.sum(X**2)))
    skewness = _safe_stat(float(skew(X)))
    kurt = _safe_stat(float(kurtosis(X)))
    p05 = _safe_stat(float(np.percentile(X, 5)))
    p25 = _safe_stat(float(np.percentile(X, 25)))
    p75 = _safe_stat(float(np.percentile(X, 75)))
    p95 = _safe_stat(float(np.percentile(X, 95)))
    iqr = _safe_stat(float(p75 - p25))
    zero_crossings = _safe_stat(float(np.sum(np.diff(np.signbit(X) != 0))))
    zero_crossing_rate = _safe_stat(float(zero_crossings / len(X)))
    return [
        mean, std, minimum, maximum, median, abs_mean, rms, peak_to_peak,
        energy, skewness, kurt, p05, p25, p75, p95, iqr, zero_crossings,
        zero_crossing_rate
    ]

def band_power(
        freqs : np.ndarray,
        power : np.ndarray,
        fmin: float,
        fmax: float,
        total_power: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return _safe_stat(float(np.sum(power[mask]) / total_power))

def frequency_domain_features(X: np.ndarray,
                              sample_rate: float) -> list[float]:
    # Placeholder for frequency domain feature extraction
    X = np.asarray(X, dtype=np.float32)
    spectrum = np.fft.rfft(X)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(X), d=1/sample_rate)
    power = magnitude**2
    total_power = float(np.sum(power)) + 1e-8
    dominant_idx = int(np.argmax(magnitude))
    dominant_freq = _safe_stat(float(freqs[dominant_idx]))
    spectral_centroid = _safe_stat(
        float(
            np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        )
    )
    low_band = band_power(freqs, power, 0.5, 3.0, total_power)
    mid_band = band_power(freqs, power, 3.0, 8.0, total_power)
    high_band = band_power(freqs, power, 8.0, 15.0,total_power)
    prob = power / total_power
    spectral_entropy = _safe_stat(float(-np.sum(prob * np.log(prob + 1e-8))))
    return [
        dominant_freq, spectral_centroid, low_band, mid_band, high_band,
        spectral_entropy
    ]

def extract_features_from_window(X: np.ndarray,
                                 sample_rate: float) -> list[float]:
    features = []
    features.extend(time_domain_features(X))
    features.extend(frequency_domain_features(X, sample_rate))
    return features

def extract_statistical_features(X: np.ndarray, sample_rate: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f'Input must be 2D array, got shape {X.shape}')
    rows = [
        extract_features_from_window(
            X = window,
            sample_rate = sample_rate,
        )
        for window in X
    ]
    return np.array(rows, dtype = np.float32)

def feature_names() -> list[str]:
    time_features = [
        'mean', 'std', 'min', 'max', 'median', 'abs_mean', 'rms',
        'peak_to_peak', 'energy', 'skewness', 'kurtosis',
        'p05', 'p25', 'p75', 'p95', 'iqr',
        'zero_crossings', 'zero_crossing_rate'
    ]
    freq_features = [
        'dominant_freq', 'spectral_centroid',
        'low_band_power', 'mid_band_power', 'high_band_power',
        'spectral_entropy'
    ]
    return time_features + freq_features