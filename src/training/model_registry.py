from __future__ import annotations
from typing import Any
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_logistic_regression(params: dict[str,Any], seed : int):
    clean = dict(params)
    clean.setdefault('class_weight', 'balanced')
    clean.setdefault('max_iter', 2000)
    clean.setdefault('random_state', seed)
    
    return Pipeline(
        steps = [
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(**clean))
        ]
    )
    
def build_random_forest(params: dict[str,Any], seed : int):
    clean = dict(params)
    clean.setdefault('random_state', seed)
    clean.setdefault('n_jobs', -1)
    return RandomForestClassifier(**clean)

def build_extra_trees(params: dict[str, Any], seed : int):
    clean = dict(params)
    clean.setdefault('random_state', seed)
    clean.setdefault('n_jobs', -1)
    return ExtraTreesClassifier(**clean)

def build_isolation_forest(params: dict[str, Any], seed : int):
    clean = dict(params)
    clean.setdefault('random_state', seed)
    clean.setdefault('n_jobs', -1)
    return IsolationForest(**clean)

MODEL_BUILDERS = {
    'logistic_regression': build_logistic_regression,
    'random_forest': build_random_forest,
    'extra_trees': build_extra_trees,
    'isolation_forest': build_isolation_forest,
}

def build_model(model_name: str, params: dict[str, Any], seed : int):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f'Model {model_name} not supported. Supported models: {list(MODEL_BUILDERS.keys())}')
    return MODEL_BUILDERS[model_name](params, seed)