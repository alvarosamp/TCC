from __future__ import annotations
import pytest
from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split, validate_full_dataset, validate_npz_keys

def test_validate_npz_keys_accepts_valid_keys(tiny_dataset_path):
    '''
    Dataset deve conter as chaves obrigatórias
    '''
    keys = validate_npz_keys(tiny_dataset_path)
    assert set(keys) == {'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'}

def test_validate_npz_keys_rejects_missing_keys(bad_dataset_missing_key_path):
    '''Dataset sem cahve aleatoria deve falhar'''
    with pytest.raises(ValueError):
        validate_npz_keys(bad_dataset_missing_key_path)

def test_load_validated_split(tiny_profile_path, tiny_dataset_path):
    '''Deve carregar os splits do dataset e validar as formas'''
    profile = PipelineProfile.load_from_yaml(tiny_profile_path)
    X_train, y_train = load_validated_split(tiny_dataset_path, 'train', profile)
    assert X_train.shape == (20, 32)
    assert y_train.shape == (20,)
    assert set(y_train.tolist()) == {0, 1}

def test_validate_full_dataset_generes_report(
        tiny_profile_path,
        tiny_dataset_path,
):
    profile = PipelineProfile.load_from_yaml(tiny_profile_path)
    report = validate_full_dataset(tiny_dataset_path, profile)
    assert 'splits' in report 
    assert set(report['splits'].keys()) == {'train', 'val', 'test'}
    assert report['splits']['train']['total'] == 20
    assert report['splits']['val']['total'] == 10
    assert report['splits']['test']['total'] == 10