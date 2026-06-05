from __future__ import annotations
from pathlib  import Path
import json
import logging
from typing import Any
import numpy as np
import tensorflow as tf
from src.core.profile import PipelineProfile
from src.core.schemas import load_validated_split
from src.core.settings import DATASET_FILE, EDGE_DIR, PROFILE_PATH, REPORTS_DIR, ensure_directories
from src.export.export_header import export_tflite_to_header

'''
Exportar o melhor modelo escolhido pelo all train
entrada : candidate_manifest
saida modelo quantizado
'''

log = logging.getLogger('export_tflite')
if not log.handlers:
    logging.basicConfig(
        level = logging.INFO,
        format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

CANDIDATE_MANIFEST = REPORTS_DIR / 'candidate_manifest.json'

def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r',encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload,
                  f,
                  indent= 2,
                  ensure_ascii = False)

def as_model_input(X: np.ndarray, model_name: str) -> np.ndarray:
    '''
    Prepara a entrada para representative dataset 
    denseae usa : (n, window_size)
    cnn/tcn/ : (n, window_size, 1)
    '''
    if model_name == 'dense_autoencoder':
        return X.astype(np.float32, copy = False)
    return X[..., np.newaxis].astype(np.float32, copy = False)

def representative_dataset_generator(X_train: np.ndarray, model_name: str, n_samples = 512):
    '''
    Gera amostras representativas para quantização
    '''
    def generator():
        for i in range(len(X_rep)):
            yield [X_rep[i:i+1]]
    return generator

def convert_float32(model: tf.keras.Model) -> bytes:
    '''Exporta tflite float32'''
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()

def convert_float16(model: tf.keras.Model) -> bytes:
    '''Exporta tflite float16'''
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()

def convert_int8(
        model : tf.keras.Model,
        X_train: np.ndarray,
        model_name: str,
    ) -> bytes:
    '''Exporta tflite int8'''
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator(X_train, model_name, n_samples= 512)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()

def export_variant(variant_name: str,
                   data: bytes,
                   model_name: str) -> dict[str, Any]:
    '''Salva um variante do tflite'''
    out_path = EDGE_DIR / f'{model_name}_{variant_name}.tflite'
    out_path.parent.mkdir(parents = True, exist_ok = True)
    out_path.write_bytes(data)
    return {
        'variant': variant_name,
        'path': str(out_path),
        'size_bytes': len(data),
        'size_kb': len(data) / 1024,
    }

def main() -> None:
    ensure_directories()
    candidate = load_json(CANDIDATE_MANIFEST)
    model_name = candidate['model_name']
    model_path = candidate['model_path']

    if not candidate.get('export_tflite', False):
        raise ValueError(f"Candidate {model_name} is not marked for TFLite export.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_path.suffix != '.h5' and model_path.suffix != '.keras':
        raise ValueError(f"Model file must be a .h5 or .keras file: {model_path}")
    
    profile = PipelineProfile.load(PROFILE_PATH)
    X_train, _ = load_validated_split(DATASET_FILE,
                                      'train',
                                      profile)
    
    log.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    exports = {}
    float32_data = convert_float32(model)
    exports['float32'] = export_variant('float32', float32_data, model_name)

    float16_data = convert_float16(model)
    exports['float16'] = export_variant('float16', float16_data, model_name)

    int8_data = convert_int8(model, X_train, model_name)
    exports['int8'] = export_variant('int8', int8_data, model_name)

    header_info = export_tflite_to_header(
        tflite_path = exports['int8']['path'],
        header_path = EDGE_DIR / f'{model_name}_int8.h',
        array_name = f'{model_name}_int8_model_data'
    )

    export_manifest = {
        'model_name': model_name,
        'exports': exports,
        'header_info': header_info,
        'source_model_path': str(model_path),
        'profile': profile.to_dict(),
        'dataset': {
            'file': str(DATASET_FILE),
            'train_samples': len(X_train),
        },
        'thresholds': candidate.get('thresholds', {}),
        'metrics': candidate.get('metrics', {}),
    }
    out_manifest = EDGE_DIR / f'{model_name}_export_manifest.json'
    save_json(export_manifest, out_manifest)
    log.info(f"Export completed. Manifest saved to {out_manifest}")
    for name, info in exports.items():
        log.info(f"Exported {name}: {info['size_kb']:.2f} KB at {info['path']}")

    log.info(f"Header file generated at {header_info['header_path']} with array name {header_info['array_name']}")

if __name__ == '__main__':
    main()

                   
