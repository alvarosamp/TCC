from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from pyexpat import model
from typing import Any
import joblib
import mlflow
import numpy as np
import optuna
import yaml
from src.core.profile import PipelineProfile
from src.core.schemas import load_validate_split
from src.core.settings import (DATASET_FILE,
                               MLFLOW_EXPERIMENT_NAME,
                               MLFLOW_TRACKING_URI,
                               MODELS_DIR,
                               PROFILE_PATH,
                               REPORTS_DIR,
                               ROOT_DIR,
                               SEED,
                               ensure_directories)
from src.features.statiscal_features import extract_statistical_features, feature_names
from src.training.evaluate import evaluate_scores, evaluate_with_validation_threshold
from src.training.hpo import suggest_params
from src.training.model_registry import build_model

log = logging.getLogger('train_all')
if not log.handlers:
    logging.basicConfig(
        level= logging.INFO,
        format = '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt = '%H:%M:%S',
    )
    
MODELS_CONFIG_PATH = ROOT_DIR / 'models.yaml'

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'YAML file not found: {path}')
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w',encoding = 'utf-8') as f:
        json.dump(data, f, indent= 2, ensure_ascii=False)
        

def enabled_models(models_cfg: dict[str, Any]) -> dict[str, Any]:
    return {name: cfg for name, cfg in models_cfg.items() if cfg.get('enabled', False)}

def supervised_scores(model, X_features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_features)[:, 1].astype(np.float32)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_features), dtype=np.float32)

    raise TypeError(f"Modelo {type(model).__name__} nao possui score supervisionado.")

def train_model_once(model_name:str,
                     family: str,
                     params: dict[str, Any],
                     X_train_f: np.ndarray,
                     y_train: np.ndarray,
                     X_val_f: np.ndarray,
                     y_val: np.ndarray,
                     profile: PipelineProfile):
    '''
    Treina um modelo e retorna os scores de validação e teste.
    '''
    if family == 'classical supervised':
        model.fit(X_train_f, y_train)
        scores_val = supervised_scores(model, X_val_f)
        
    elif family == 'classical unsupervised':
        normal_mask = y_train == profile.normal_label
        model.fit(X_train_f[normal_mask])
        scores_val = -model.decision_function(X_val_f)
    
    else:
        raise ValueError(f'Unknown model family: {family}')
    
    val_metrics = evaluate_scores(y_true = y_val, scores = scores_val, threshold = None)
    return model, scores_val, val_metrics

def tune_model(
    model_name: str,
    model_cfg: dict[str, Any],
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_val_f: np.ndarray,
    y_val: np.ndarray,
    profile: PipelineProfile,
    selection_metric: str,
) -> dict[str, Any]:
    family = model_cfg['family']
    base_params = dict(model_cfg.get('params', {}))
    tune_cfg = dict(model_cfg.get('tune', None))
    search_space = dict(tune_cfg.get('search_space', {}))
    n_trials = int(tune_cfg.get('n_trials', 20))

    if not tune_cfg.get('enabled', False) or not search_space:
        return {
            'used_optuna': False,
            'best_params': base_params,
            'best_value': None,
            'n_trials': 0,
        }    
    def objective(trial: optuna.Trial) -> float:
        trial_params = dict(base_params)
        trial_params.update(
            suggest_params(trial = trial, search_space = search_space)
        )
        _, _, val_metrics = train_model_once(
            model_name = model_name,
            family = family,
            params = trial_params,
            X_train_f = X_train_f,
            y_train = y_train,
            X_val_f = X_val_f,
            y_val = y_val,
            profile = profile,
        )
        return val_metrics[selection_metric]
    
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = n_trials)
    best_params = dict(base_params)
    best_params.update(study.best_params)
    return {
        'used_optuna': True,
        'best_params': best_params,
        'best_value': study.best_value,
        'n_trials': n_trials,
    }

def train_final_model(
    model_name : str,
    model_cfg : dict[str, Any],
    params: dict[str, Any],
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_val_f : np.ndarray,
    y_val : np.ndarray,
    X_test_f : np.ndarray,
    y_test : np.ndarray,
    profile : PipelineProfile,
) -> dict[str, Any]:
    '''
    Treina o modelo final com os melhores parametros encontrados e avalia no teste.
    '''
    family = model_cfg['family']
    model = build_model(model_name, params, SEED)
    
    if family == 'classical_supervised':
        model.fit(X_train_f, y_train)
        scores_val = supervised_scores(model, X_val_f)
        scores_test = supervised_scores(model, X_test_f)
        
    elif family =='classical_unsupervised':
        normal_mask = y_train == profile.normal_label
        model.fit(X_train_f[normal_mask])
        scores_val = -model.decision_function(X_val_f)
        scores_test = -model.decision_function(X_test_f)
    else:
        raise ValueError(f'Unknown model family: {family}')
    
    evaluation = evaluate_with_validation_threshold(
        y_val = y_val,
        scores_val = scores_val,
        y_test = y_test,
        scores_test = scores_test,
    )
    model_path = MODELS_DIR / f'{model_name}.joblib'
    joblib.dump(model, model_path)
    return {
        'model_path': str(model_path),
        'evaluation': evaluation,
        'model' : evaluation,
    }
    
def log_result_to_mlflow(result: dict[str, Any]) -> None:
    model_name = result['model_name']
    evaluation = result['evaluation']
    metrics_path = REPORTS_DIR / f'{model_name}_metrics.json'
    save_json(metrics_path, evaluation)
    
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('family', result['family'])
    mlflow.log_param('profile', result['profile']['profile_name'] + ':' + result['profile']['split_name'])
    mlflow.log_param('dataset', result['dataset'])
    mlflow.log_param('edge_candidate', result['edge_candidate'])
    mlflow.log_param('export_tflite', result['export_tflite'])
    mlflow.log_param('priority', result['priority'])
    mlflow.log_param('used_optuna', result['used_optuna'])
    
    for key, value in result['params'].items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)
            
    mlflow.log_metric('val_roc_auc', evaluation['val']['roc_auc'])
    mlflow.log_metric('val_f1', evaluation['val']['f1'])
    mlflow.log_metric('test_roc_auc', evaluation['test']['roc_auc'])
    mlflow.log_metric('test_f1', evaluation['test']['f1'])
    mlflow.log_metric('threshold_from_val', evaluation['threshold_from_val'])
    
    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(result['model_path'])
    
def metric_value(result: dict[str,Any], split: str, metric: str) -> float:
    return float(result['evaluation'][split][metric])

def select_best_model(
    results: list[dict[str, Any]],
    selection_cfg : dict[str, Any],
) -> dict[str, Any]:
    metric = selection_cfg.get('metric', 'auc_pr')
    split = selection_cfg.get('split', 'test')
    mode = selection_cfg.get('mode', 'maximize')
    if not results:
        raise ValueError('No results to select from.')
    
    reverse = mode == 'maximize'
    return sorted(
        results,
        key = lambda r: metric_value(r, split, metric),
        reverse = reverse,
    )[0]
    
def save_comparison_report(
    results: list[dict[str, Any]],
    best: dict[str, Any],
    selection_cfg: dict[str, Any],
) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "model_name": r["model_name"],
                "family": r["family"],
                "priority": r["priority"],
                "edge_candidate": r["edge_candidate"],
                "export_tflite": r["export_tflite"],
                "used_optuna": r["hpo"]["used_optuna"],
                "val_auc_pr": r["evaluation"]["val"]["auc_pr"],
                "val_f1": r["evaluation"]["val"]["f1"],
                "test_auc_pr": r["evaluation"]["test"]["auc_pr"],
                "test_f1": r["evaluation"]["test"]["f1"],
                "threshold_from_val": r["evaluation"]["threshold_from_val"],
                "model_path": r["model_path"],
            }
        )
    csv_path = REPORTS_DIR / 'model_comparison.csv'
    md_path = REPORTS_DIR / 'model_comparison.md'
    json_path = REPORTS_DIR / 'model_comparison.json'
    candidate_path = REPORTS_DIR / 'model_candidate.json'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldname=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Modelo | Familia | AUC-PR Test | F1 Test | Edge | Optuna |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['model_name']} "
                f"| {row['family']} "
                f"| {row['test_auc_pr']:.4f} "
                f"| {row['test_f1']:.4f} "
                f"| {row['edge_candidate']} "
                f"| {row['used_optuna']} |\n"
            )

    save_json(
        json_path,
        {
            "selection": selection_cfg,
            "results": results,
            "best_model": best["model_name"],
        },
    )

    save_json(
        candidate_path,
        {
            "selected_by": selection_cfg,
            "model_name": best["model_name"],
            "family": best["family"],
            "priority": best["priority"],
            "edge_candidate": best["edge_candidate"],
            "export_tflite": best["export_tflite"],
            "profile": best["profile"],
            "dataset": best["dataset"],
            "model_path": best["model_path"],
            "threshold": best["evaluation"]["threshold_from_val"],
            "metrics": best["evaluation"],
            "hpo": best["hpo"],
            "params": best["params"],
        },
    )

    log.info(f"Comparacao CSV: {csv_path}")
    log.info(f"Comparacao MD:  {md_path}")
    log.info(f"Candidate:      {candidate_path}")
    
    
def main() -> None:
    ensure_directories()
    profile = PipelineProfile.load_from_yaml(PROFILE_PATH)
    models_cfg = load_yaml(MODELS_CONFIG_PATH)
    models_to_train = enabled_models(models_cfg)
    selection_cfg = models_cfg.get('selection', {})
    log.info('=' * 60)
    log.info('Carregando dataset')
    log.info('=' * 60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_validate_split(DATASET_FILE, profile)
    log.info(f'Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features')
    
    log.info('=' * 60)
    log.info('Extraindo features classicas')
    X_train_f = extract_statistical_features(X_train, profile.sampling_rate)
    X_val_f = extract_statistical_features(X_val, profile.sampling_rate)
    X_test_f = extract_statistical_features(X_test, profile.sampling_rate)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    all_results = []
    
    for model_name, model_cfg in models_to_train.items():
        family = model_cfg.get('family')
        if family not in ['classical_supervised', 'classical_unsupervised']:
            log.warning(f'Modelo {model_name} tem familia desconhecida: {family}. Pulando.')
            continue
        
        log.info('=' * 60)
        log.info(f'Treinando modelo: {model_name} (familia: {family})')
        log.info('=' * 60)
        
        with mlflow.start_run(run_name = model_name):
            hpo_result = tune_model(
                model_name = model_name,
                model_cfg = model_cfg,
                X_train_f = X_train_f,
                y_train = y_train,
                X_val_f = X_val_f,
                y_val = y_val,
                profile = profile,
                selection_metric = selection_cfg.get('metric', 'auc_pr'),
            )
            final = train_final_model(
                model_name = model_name,
                model_cfg = model_cfg,
                params = hpo_result['best_params'],
                X_train_f = X_train_f,
                y_train = y_train,
                X_val_f = X_val_f,
                y_val = y_val,
                X_test_f = X_test_f,
                y_test = y_test,
                profile = profile,
            )
            result = {
                "model_name": model_name,
                "family": family,
                "priority": model_cfg.get("priority", "baseline"),
                "edge_candidate": bool(model_cfg.get("edge_candidate", False)),
                "export_tflite": bool(model_cfg.get("export_tflite", False)),
                "profile": profile.to_dict(),
                "dataset": str(DATASET_FILE),
                "model_path": final["model_path"],
                "params": hpo_result["best_params"],
                "hpo": hpo_result,
                "feature_names": feature_names(),
                "evaluation": final["evaluation"],
            }
            log_result_to_mlflow(result)
            all_results.append(result)
            
            log.info(
                f'{model_name} : '
                f'Val AUC-PR: {final["evaluation"]["val"]["auc_pr"]:.4f}, '
                f'test_f1 = {final["evaluation"]["test"]["f1"]:.4f}, '
            )
    
    best = select_best_model(
        results = all_results,
        selection_cfg = selection_cfg,
    )
    save_comparison_reports(
        results = all_results,
        best = best,
        selection_cfg = selection_cfg,
    )
    log.info("=" * 80)
    log.info("MELHOR MODELO")
    log.info("=" * 80)
    log.info(f"Modelo: {best['model_name']}")
    log.info(f"Familia: {best['family']}")
    log.info(f"AUC-PR test: {best['evaluation']['test']['auc_pr']:.4f}")
    log.info(f"F1 test:     {best['evaluation']['test']['f1']:.4f}")


if __name__ == "__main__":
    main()  