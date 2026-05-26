from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc_seismic.data import load_available_splits, normal_only
from tcc_seismic.features import extract_statistical_features
from tcc_seismic.io import write_json
from tcc_seismic.metrics import evaluate_with_validation_threshold
from tcc_seismic.paths import results_dir


def require_sklearn_models():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn joblib"
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "IsolationForest": IsolationForest,
        "LocalOutlierFactor": LocalOutlierFactor,
        "LogisticRegression": LogisticRegression,
        "OneClassSVM": OneClassSVM,
        "RandomForestClassifier": RandomForestClassifier,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def score_unsupervised(model, X_features: np.ndarray) -> np.ndarray:
    if hasattr(model, "score_samples"):
        return -model.score_samples(X_features)
    if hasattr(model, "decision_function"):
        return -model.decision_function(X_features)
    raise TypeError(f"Model {type(model).__name__} does not expose anomaly scores")


def score_supervised(model, X_features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_features)
        return np.asarray(scores, dtype=float)
    raise TypeError(f"Model {type(model).__name__} does not expose supervised scores")


def build_unsupervised_models(args, sk):
    return {
        "isolation_forest": sk["IsolationForest"](
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            random_state=args.seed,
        ),
        "one_class_svm": sk["make_pipeline"](
            sk["StandardScaler"](),
            sk["OneClassSVM"](kernel="rbf", nu=args.contamination, gamma="scale"),
        ),
        "local_outlier_factor": sk["LocalOutlierFactor"](
            n_neighbors=20,
            contamination=args.contamination,
            novelty=True,
        ),
    }


def build_supervised_models(args, sk):
    return {
        "logistic_regression": sk["make_pipeline"](
            sk["StandardScaler"](),
            sk["LogisticRegression"](class_weight="balanced", max_iter=2000),
        ),
        "random_forest": sk["RandomForestClassifier"](
            n_estimators=args.n_estimators,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "extra_trees": sk["ExtraTreesClassifier"](
            n_estimators=args.n_estimators,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model, X_val_f, y_val, X_test_f, y_test, supervised: bool, strategy: str):
    if supervised:
        scores_val = score_supervised(model, X_val_f)
        scores_test = score_supervised(model, X_test_f)
    else:
        scores_val = score_unsupervised(model, X_val_f)
        scores_test = score_unsupervised(model, X_test_f)

    return {
        "model": name,
        "mode": "supervised" if supervised else "unsupervised_anomaly",
        "evaluation": evaluate_with_validation_threshold(
            y_val,
            scores_val,
            y_test,
            scores_test,
            strategy=strategy,
        ),
    }


def train_one_split(split, args) -> list[dict]:
    sk = require_sklearn_models()
    X_train_f = extract_statistical_features(split.X_train, sample_rate=args.sample_rate)
    X_val_f = extract_statistical_features(split.X_val, sample_rate=args.sample_rate)
    X_test_f = extract_statistical_features(split.X_test, sample_rate=args.sample_rate)

    results = []

    for name, model in build_unsupervised_models(args, sk).items():
        model.fit(extract_statistical_features(normal_only(split.X_train, split.y_train), args.sample_rate))
        result = evaluate_model(
            name,
            model,
            X_val_f,
            split.y_val,
            X_test_f,
            split.y_test,
            supervised=False,
            strategy=args.threshold_strategy,
        )
        result["split"] = split.name
        results.append(result)

    if len(np.unique(split.y_train)) >= 2:
        for name, model in build_supervised_models(args, sk).items():
            model.fit(X_train_f, split.y_train)
            result = evaluate_model(
                name,
                model,
                X_val_f,
                split.y_val,
                X_test_f,
                split.y_test,
                supervised=True,
                strategy=args.threshold_strategy,
            )
            result["split"] = split.name
            results.append(result)
    else:
        results.append(
            {
                "split": split.name,
                "mode": "supervised",
                "skipped": True,
                "reason": "y_train has only one class; supervised ML needs normal and event samples in train.",
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classical ML baselines with explainable time/frequency features."
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "legacy_v2", "legacy_v1", "v3"],
        default="auto",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["best_f1", "percentile_normal", "target_recall"],
        default="best_f1",
    )
    parser.add_argument("--sample-rate", type=float, default=40.0)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_available_splits(args.dataset)
    results = []
    for split in splits:
        results.extend(train_one_split(split, args))

    out_dir = results_dir("models_corrected", "classical_ml")
    write_json(out_dir / "summary.json", {"results": results})
    print(json.dumps({"results_dir": str(out_dir), "n_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()

