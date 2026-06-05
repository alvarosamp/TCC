from __future__ import annotations
import numpy as np
from src.features.statistical_features import extract_statistical_features, feature_names

def test_extract_statistical_features_shape():
    '''Deve retornar o número correto de features'''
    rng = np.random.default_rng(42)
    X = rng.normal(size = (5,32)).astype(np.float32)
    features = extract_statistical_features(X)
    