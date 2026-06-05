from __future__ import annotations
import pytest
from src.core.profile import PipelineProfile, infer_window_size 

def test_load_profile_from_yaml(tiny_profile_path):
    '''Deve carregar corretament o yaml'''
    profile = PipelineProfile.load_from_yaml(tiny_profile_path)
    assert profile.profile_name == 'test_profile_v1'
    assert profile.profile_version == '1.0.0'
    assert profile.window_size == 32
    assert profile.normal_label == 0
    assert profile.anomaly_label == 1
    assert profile.preprocessing['offline_pipeline'] == ['zscore_per_window']

def test_infer_window_size_accepts_commom_shape():
    assert infer_window_size((100,32)) == 32
    assert infer_window_size((100,32,1)) == 32
    assert infer_window_size((100, 1,32)) ==32

def test_profile_rejects_window_shape(tiny_profile_path):
    '''Se o dataset tiver janela errada, o erro deve vir cedo'''
    profile = PipelineProfile.load_from_yaml(tiny_profile_path)
    with pytest.raises(ValueError):
        profile.validate_window_size((10, 64))