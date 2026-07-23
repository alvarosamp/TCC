"""
Benchmark cross-domain: compara o pipeline em múltiplos domínios.

Uso:
  python scripts/benchmark_cross_domain.py
  python scripts/benchmark_cross_domain.py --domains seismic 3w
  python scripts/benchmark_cross_domain.py --models tiny_tcn random_forest
  python scripts/benchmark_cross_domain.py --output results/cross_domain_benchmark

Fluxo por domínio:
  1. Carrega dataset .npz processado
  2. Treina modelos configurados
  3. Avalia com métricas completas: AUC-PR, VUS-PR, PA-F1, Event-F1, FP/h
  4. Salva resultados por domínio
  5. Gera tabela comparativa CSV + JSON + figura

Saída:
  results/cross_domain_benchmark.csv
  results/cross_domain_benchmark.json
  results/cross_domain_benchmark.png
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("benchmark_cross_domain")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ============================================================
# CONFIGURAÇÃO DOS DOMÍNIOS
# ============================================================

# Cada domínio tem: dataset_path, profile_path, n_channels, sampling_rate, window_seconds
DOMAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "seismic": {
        "dataset_path": "data/processed/seismic_edge_v1/dataset.npz",
        "profile_path": "profiles/seismic_edge_v1.json",
        "n_channels": 1,
        "sampling_rate": 40.0,
        "window_seconds": 20.0,
        "description": "Deteccao sismica (MiniSEED, 1 canal, 40 Hz)",
    },
    "3w": {
        "dataset_path": "data/processed/3w/dataset.npz",
        "profile_path": "profiles/3w_edge_v1.json",
        "n_channels": 8,
        "sampling_rate": 1.0,
        "window_seconds": 60.0,
        "description": "Pocas de petroleo Petrobras 3W (8 sensores, 1 Hz)",
    },
}

DEFAULT_MODELS = ["random_forest", "extra_trees", "tiny_cnn", "tiny_tcn"]
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256


# ============================================================
# TREINAMENTO LEVE (sem MLflow, sem HPO) PARA BENCHMARKING
# ============================================================

def _load_split(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    return (
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        data["X_test"],  data["y_test"],
    )


def _prepare_neural_input(X: np.ndarray, n_channels: int) -> np.ndarray:
    """Garante shape (N, window_size, n_channels) para redes neurais."""
    if X.ndim == 2:
        return X[:, :, np.newaxis]
    return X


def _prepare_classical_input(X: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Extrai features 2D para modelos clássicos."""
    from src.features.statistical_features import extract_statistical_features
    return extract_statistical_features(X, sampling_rate)


def _fp_per_hour(y_true: np.ndarray, y_pred: np.ndarray, window_seconds: float) -> float:
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    total_hours = len(y_true) * window_seconds / 3600.0
    return float(fp / (total_hours + 1e-8))


def _train_classical(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_rate: float,
) -> Any:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    X_tr = _prepare_classical_input(X_train, sampling_rate)
    params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}

    if model_name == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model_name == "extra_trees":
        clf = ExtraTreesClassifier(**params)
    elif model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Modelo classico desconhecido: {model_name}")

    clf.fit(X_tr, y_train)
    return clf


def _train_neural(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_channels: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Any:
    import tensorflow as tf
    from src.training.neural_models import build_neural_model

    window_size = X_train.shape[1]
    X_tr = _prepare_neural_input(X_train, n_channels)
    X_v = _prepare_neural_input(X_val, n_channels)

    params = {
        "n_blocks": 4,
        "filters": 16,
        "kernel_size": 7,
        "dense_units": 24,
        "dropout": 0.15,
        "learning_rate": 0.001,
    }

    model = build_neural_model(model_name, window_size, params, n_channels=n_channels)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            patience=5,
            restore_best_weights=True,
            mode="max",
        )
    ]

    model.fit(
        X_tr, y_train,
        validation_data=(X_v, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=0,
    )
    return model


def _score_classical(clf: Any, X: np.ndarray, sampling_rate: float) -> np.ndarray:
    X_feat = _prepare_classical_input(X, sampling_rate)
    return clf.predict_proba(X_feat)[:, 1].astype(np.float32)


def _score_neural(model: Any, X: np.ndarray, n_channels: int) -> np.ndarray:
    X_in = _prepare_neural_input(X, n_channels)
    return model.predict(X_in, verbose=0).squeeze().astype(np.float32)


# ============================================================
# AVALIAÇÃO
# ============================================================

def _evaluate_domain_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    domain_cfg: dict[str, Any],
) -> dict[str, Any]:
    from src.training.evaluate import evaluate_scores

    n_channels = domain_cfg["n_channels"]
    sampling_rate = domain_cfg["sampling_rate"]
    window_seconds = domain_cfg["window_seconds"]

    is_classical = model_name in {"random_forest", "extra_trees", "logistic_regression"}
    is_neural = model_name in {"tiny_cnn", "tiny_tcn", "lstm_classifier"}

    t0 = time.time()
    try:
        if is_classical:
            clf = _train_classical(model_name, X_train, y_train, sampling_rate)
            scores_val = _score_classical(clf, X_val, sampling_rate)
            scores_test = _score_classical(clf, X_test, sampling_rate)
            n_params = None
        elif is_neural:
            model = _train_neural(
                model_name, X_train, y_train, X_val, y_val, n_channels
            )
            scores_val = _score_neural(model, X_val, n_channels)
            scores_test = _score_neural(model, X_test, n_channels)
            n_params = int(model.count_params())
        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")
    except Exception as e:
        log.warning(f"  Falha ao treinar {model_name}: {e}")
        return {"model": model_name, "error": str(e)}

    train_time = time.time() - t0

    # threshold pela validação, aplica no teste
    val_metrics = evaluate_scores(y_val, scores_val)
    threshold = val_metrics["threshold"]
    test_metrics = evaluate_scores(y_test, scores_test, threshold=threshold)

    y_pred_test = (scores_test >= threshold).astype(np.int32)
    fp_h = _fp_per_hour(y_test, y_pred_test, window_seconds)

    result: dict[str, Any] = {
        "model": model_name,
        "auc_pr_val": round(val_metrics["auc_pr"], 4),
        "auc_pr_test": round(test_metrics["auc_pr"], 4),
        "auc_roc_test": round(test_metrics["auc_roc"], 4),
        "f1_test": round(test_metrics["f1"], 4),
        "pa_f1_test": round(test_metrics["pa_f1"], 4),
        "event_f1_test": round(test_metrics["event_f1"], 4),
        "vus_pr_test": round(test_metrics["vus_pr"], 4),
        "fp_per_hour": round(fp_h, 3),
        "threshold": round(threshold, 4),
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "edge_capable": is_neural,
    }
    return result


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def run_benchmark(
    domains: list[str],
    models: list[str],
    output_prefix: Path,
    root_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for domain_name in domains:
        if domain_name not in DOMAIN_CONFIGS:
            log.warning(f"Dominio '{domain_name}' nao configurado. Disponiveis: {list(DOMAIN_CONFIGS)}")
            continue

        cfg = DOMAIN_CONFIGS[domain_name]
        dataset_path = root_dir / cfg["dataset_path"]

        if not dataset_path.exists():
            log.warning(
                f"Dataset nao encontrado: {dataset_path}\n"
                f"  Execute o adapter do dominio '{domain_name}' antes do benchmark."
            )
            continue

        log.info(f"\n{'='*60}")
        log.info(f"DOMINIO: {domain_name} — {cfg['description']}")
        log.info(f"{'='*60}")

        try:
            X_train, y_train, X_val, y_val, X_test, y_test = _load_split(dataset_path)
        except Exception as e:
            log.error(f"Erro ao carregar {dataset_path}: {e}")
            continue

        log.info(f"  Treino:  {len(X_train)} janelas  (shape={X_train.shape})")
        log.info(f"  Val:     {len(X_val)} janelas")
        log.info(f"  Teste:   {len(X_test)} janelas")
        log.info(f"  Anomaly ratio (teste): {(y_test==1).mean():.1%}")

        for model_name in models:
            log.info(f"\n  > {model_name} ...")
            result = _evaluate_domain_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                domain_cfg=cfg,
            )
            result["domain"] = domain_name

            if "error" not in result:
                log.info(
                    f"    AUC-PR={result['auc_pr_test']:.4f}  "
                    f"VUS-PR={result['vus_pr_test']:.4f}  "
                    f"PA-F1={result['pa_f1_test']:.4f}  "
                    f"Event-F1={result['event_f1_test']:.4f}  "
                    f"FP/h={result['fp_per_hour']:.3f}"
                )
            else:
                log.warning(f"    Erro: {result['error']}")

            all_results.append(result)

    if not all_results:
        log.error("Nenhum resultado gerado. Verifique os datasets e modelos.")
        return

    # ── CSV ────────────────────────────────────────────────
    csv_path = output_prefix.with_suffix(".csv")
    import csv as csv_mod
    fieldnames = [
        "domain", "model",
        "auc_pr_val", "auc_pr_test", "auc_roc_test",
        "f1_test", "pa_f1_test", "event_f1_test", "vus_pr_test",
        "fp_per_hour", "threshold", "n_params", "train_time_s", "edge_capable", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    log.info(f"\nCSV salvo em {csv_path}")

    # ── JSON ───────────────────────────────────────────────
    json_path = output_prefix.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"JSON salvo em {json_path}")

    # ── Figura ────────────────────────────────────────────
    _plot_results(all_results, output_prefix.with_suffix(".png"), domains, models)


def _plot_results(
    results: list[dict[str, Any]],
    output_path: Path,
    domains: list[str],
    models: list[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib nao disponivel. Figura nao gerada.")
        return

    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    metrics = ["auc_pr_test", "vus_pr_test", "pa_f1_test", "event_f1_test"]
    metric_labels = ["AUC-PR", "VUS-PR", "PA-F1", "Event-F1"]
    n_metrics = len(metrics)
    n_domains = len(domains)

    fig, axes = plt.subplots(n_domains, n_metrics, figsize=(5 * n_metrics, 4 * n_domains))
    if n_domains == 1:
        axes = axes[np.newaxis, :]

    for di, domain in enumerate(domains):
        domain_results = [r for r in valid if r.get("domain") == domain]
        if not domain_results:
            continue

        model_names = [r["model"] for r in domain_results]

        for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[di, mi]
            values = [r.get(metric, 0.0) for r in domain_results]
            colors = ["#2ecc71" if r.get("edge_capable") else "#3498db" for r in domain_results]
            bars = ax.barh(model_names, values, color=colors)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel(mlabel)
            if mi == 0:
                ax.set_ylabel(f"Dominio: {domain}")
            ax.set_title(f"{mlabel} — {domain}")
            for bar, val in zip(bars, values):
                ax.text(
                    float(bar.get_width()) + 0.01,
                    float(bar.get_y()) + float(bar.get_height()) / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=8,
                )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Edge (neural)"),
        Patch(facecolor="#3498db", label="Nao-edge (classico)"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", ncol=2)
    fig.suptitle("Benchmark Cross-Domain — Métricas de Avaliação", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva em {output_path}")


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cross-domain de deteccao de anomalias")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_CONFIGS.keys()),
        help=f"Dominios a avaliar. Disponiveis: {list(DOMAIN_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Modelos a avaliar",
    )
    parser.add_argument(
        "--output",
        default="results/cross_domain_benchmark",
        help="Prefixo do arquivo de saida (sem extensao)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs para modelos neurais",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Pasta raiz do projeto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        domains=args.domains,
        models=args.models,
        output_prefix=Path(args.output),
        root_dir=Path(args.root),
        epochs=args.epochs,
    )
