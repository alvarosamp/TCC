"""1. carregar config.yaml
2. carregar o profile ativo
3. carregar o dataset NPZ
4. validar chaves
5. validar shape das janelas
6. validar labels
7. gerar relatório JSON"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
from src.core.profile import PipelineProfile
from src.core.schemas import validate_full_dataset
from src.core.schemas import DATASET_FILE, PROFILE_PATH, REPORTS_DIR, ensure_directories, print_settings

# OBJETIVO DO ARQUIVO
# ============================================================
#
# Este script valida se o dataset esta no contrato generico:
#
#   X_train, y_train
#   X_val, y_val
#   X_test, y_test
#
# Ele deve ser executado antes de qualquer treino.
#
# Se esse script falhar, o treino nao deve comecar.
#
# Isso segue a ideia de sistemas de IA:
#   dados precisam de contrato e validacao explicita.

log = logging.getLogger('validate_dataset')
if not log.handlers:
    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    

def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)
        
        
def main():
    '''Valida o dataset e gera um relatorio JSON'''
    ensure_directories()
    print_settings()
    log.info("Carregando profile...")
    log.info(f"Profile path: {PROFILE_PATH}")
    profile = PipelineProfile.from_yaml(PROFILE_PATH)
    report = validate_full_dataset(profile, DATASET_FILE)
    output_path = REPORTS_DIR / f"{profile.full_version}_validation_report.json"
    save_json(output_path, report)
    log.info(f"Relatório de validação salvo em: {output_path}")
    
    for split_name, split_info in report["splits"].items():
        log.info(
            f"{split_name}: "
            f"total={split_info['total']} "
            f"normal={split_info['normal']} "
            f"anomaly={split_info['anomaly']} "
            f"baseline_auc_pr={split_info['baseline_auc_pr']:.4f} "
            f"shape={split_info['x_shape']}"
        )

    log.info("=" * 80)
    log.info("DATASET VALIDADO COM SUCESSO")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
