from __future__ import annotations
import os 
from pathlib import Path
import yaml

#Raiz do projeto
# parents[2] de settings.py: src/core -> src -> versao_final
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_ROOT  # alias usado pelos modulos de treino
CONFIG_PATH = PROJECT_ROOT / "config" / "configs" / "config.yaml"

#Lendo config
def load_yaml(path: str | Path) -> dict:
    '''Le um yaml e retorna um dicionário'''
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with open(path, 'r', encoding = 'utf-8') as f: 
        return yaml.safe_load(f)
    
CFG = load_yaml(CONFIG_PATH)


#Config geral
PROJECT_NAME = CFG['project']['name']
SEED = int(CFG['project'].get('seed', 42))

#caminhos
DATA_BASE = Path(os.environ.get('TCC_DATA_BASE', CFG['paths']['data_base']))
RAW_DIR = Path(os.environ.get('TCC_RAW_DIR', CFG['paths']['raw_dir']))
PROCESSED_DIR = Path(os.environ.get('TCC_PROCESSED_DIR', CFG['paths']['processed_dir']))
ARTEFACTS_DIR = PROJECT_ROOT / CFG['paths']['artefacts_dir']
REPORTS_DIR = ARTEFACTS_DIR / CFG['paths']['reports_dir']
MODELS_DIR = ARTEFACTS_DIR / CFG['paths']['models_dir']
EDGE_DIR = ARTEFACTS_DIR / CFG['paths']['edge_dir']
RUNS_DIR = ARTEFACTS_DIR / CFG['paths']['runs_dir']

#profile
PROFILE_NAME = CFG['profile']['name']
PROFILE_PATH = PROJECT_ROOT / CFG['profile']['path']

#dataset
SPLIT_NAME = CFG['dataset']['split_name']
DATASET_FILE = PROCESSED_DIR / CFG['dataset']['file_name']

#Mlflow
MLFLOW_EXPERIMENT_NAME = CFG['mlflow']['experiment_name']
MLFLOW_TRACKING_URI = CFG['mlflow']['tracking_uri']

#Quality gate
QUALITY_GATE = CFG['quality_gate']

#OTA
OTA_CONFIG = CFG.get('ota', {})

#funcoes auxiliares
def ensure_directories() -> Path:
    '''cria os diretorios principais de saida'''
    for path in [
        PROCESSED_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        EDGE_DIR,
        RUNS_DIR,
        ARTEFACTS_DIR
    ]:
        path.mkdir(parents=True, exist_ok=True)

def print_settings() -> None:
    '''imprime as principais configurações'''
    print(f"Project: {PROJECT_NAME}"
          f"\nProfile: {PROFILE_NAME}"
          f"\nDataset: {DATASET_FILE}"
          f"\nMLflow experiment: {MLFLOW_TRACKING_URI}"
          f"\nQuality gate: {QUALITY_GATE}"
          f"\nProject root: {PROJECT_ROOT}"
          f"\nData base: {DATA_BASE}")
    