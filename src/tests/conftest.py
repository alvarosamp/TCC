'''
Define fixtures reutilizaveis

'''
from __future__ import annotations
import json 
import sys
from anyio import Path
import pytest
import yaml
import numpy as np
ROOT_DIR = PATH(__file__).resolve.parents[1]
SRC_DIR = ROOT_DIR / 'src'

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

@pytest.fixture #pytest fixture para carregar o arquivo de configuração
def tiny_profile_dict() -> dict:
    return {
        'profile_name': 'test_profile_v1',
        'profile_version': '1.0.0',
        'task': ' binary_anomaly_detection',
        'domain': 'test_domain',
        'description': 'This is a test profile for binary anomaly detection.',
        'sampling_rate': 40.0,
        'window_size': 32,
        'window_seconds' : 0.8,
        'step_seconds' : 0.4,
        'overlap' : 0.5,
        'normal_label': 0,
        'anomaly_label': 1,
        'normal_name': 'normal',
        'anomaly_name': 'anomaly',
        'split_name': 'evento',
        'primary_metric': 'auc_pr',
        'secondary_metrics': ['auc_roc', 'f1', 'precision', 'recall'],
        'preprocessing' : {
            'offline_pipeline': [
                'zscore_per_window',
            ],
            'edge_pipeline': [
                'zscore_per_window',
            ],
        'remove_response_offline': False,
        'remove_response_embedded': False,
        'stationxml_required': False,
        'bandpass_hz' : None,
        'normalization': 'zscore_per_window',
        },
        'embedded':{
            'target': 'esp32',
            'runtime': 'tensorflow_lite_micro',
            'decision_interval_ms': 10000,
            'preprocessing_version': 'test_preproc_v1',
            'ota_strategy_initial': 'firmware_full_image',
            'ota_strategy_future': 'separete_model_partition',
        },
    }

@pytest.fixture
def tiny_profile_json( tmp_path: Path, tiny_profile_dict: dict) -> Path:
    path = tmp_path / 'tiny_profile.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tiny_profile_dict, f, sort_keys= False, allow_unicode=True, indent=4)

    return path

@pytest.fixture
def tiny_dataset_path(tmp_path: Path) -> Path:
    '''Cria um dataset com base no do projeto'''
    rng = np.random.default_rng(seed=42)
    window_size = 32
    X_train = rng.normal(0, 1, size = (20, window_size)).astype(np.float32)
    y_train = np.array([0]*14 + [1]*10, dtype=np.int8)
    X_val = rng.normal(0, 1, size = (10, window_size)).astype(np.float32)
    y_val = np.array([0]*5 + [1]*5, dtype=np.int8)
    X_test = rng.normal(0, 1, size = (10, window_size)).astype(np.float32)
    y_test = np.array([0]*5 + [1]*5, dtype=np.int8)

    path = tmp_path / 'tiny_dataset.npz'
    np.savez_compressed(
        path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )
    return path

@pytest.fixture()
def bad_dataset_missing_key_path(tmp_path: Path) -> Path:
    ''' Dataset invalido de forma proposital'''
    rng = np.random.default_rng(seed=42)
    path = tmp_path / 'bad_dataset.npz'
    np.savez_compressed(
        path,
        X_train=rng.normal(0, 1, size = (20, 32)).astype(np.float32),
        y_train=np.array([0]*14 + [1]*10, dtype=np.int8),
        X_val=rng.normal(0, 1, size = (10, 32)).astype(np.float32),
        y_val=np.array([0]*5 + [1]*5, dtype=np.int8),
        X_test=rng.normal(0, 1, size = (10, 32)).astype(np.float32),
        # X_test e y_test estão faltando
    )
    return path
