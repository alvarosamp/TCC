from __future__ import annotations
from typing import Any
import optuna

def suggest_params(trial: optuna.Trial,
                   search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Sugere parametros para um trial do Optuna.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec['type']
        
        if kind == 'int':
            params[name] = trial.suggest_int(
                name,
                int(spec['low']),
                int(spec['high']),
            )
        elif kind == 'float': 
            params[name] = trial.suggest_float(
                name,
                float(spec['low']),
                float(spec['high']),
                log = bool(spec.get('log', False)),
            )
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(
                name,
                spec['choices'],
            )
            
        else:
            raise ValueError(f'Unknown parameter kind: {kind}')
    return params