"""
Etapa 2 - Checagem de drift de dados.

Compara um lote novo de dados contra a referência estatística criada no treino.

Entrada:
  - artefacts/monitoring/drift_reference.json
  - dataset NPZ com X_train/X_val/X_test
  - split a ser analisado: train, val ou test

Saída:
  - artefacts/monitoring/drift_report.json

Métricas usadas:
  - z-score shift: quanto a média atual mudou em desvios-padrão da referência
  - PSI: Population Stability Index
  - KS test: compara distribuições entre referência e lote atual

Interpretação:
  - drift baixo: sistema saudável
  - drift moderado: observar
  - drift alto: candidato a retreino
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from src.features.statistical_features import (
    extract_statistical_features,
    feature_names,
)


DEFAULT_REFERENCE_PATH = Path("artefacts/monitoring/drift_reference.json")
DEFAULT_OUTPUT_PATH = Path("artefacts/monitoring/drift_report.json")


def load_npz_split(dataset_path: Path, split: str) -> np.ndarray:
    """Carrega X_split de um arquivo NPZ."""
    data = np.load(dataset_path)
    key = f"X_{split}"

    if key not in data.files:
        raise KeyError(
            f"Chave {key} nao encontrada em {dataset_path}. "
            f"Chaves disponiveis: {data.files}"
        )

    return data[key]


def compute_feature_matrix(X: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Converte janelas temporais em matriz de features estatísticas.

    Entrada:
      X: shape (n_amostras, janela)

    Saída:
      features: shape (n_amostras, n_features)
    """
    return extract_statistical_features(X, sample_rate=sample_rate)


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcula Population Stability Index.

    PSI mede quanto a distribuição atual mudou em relação à referência.

    Regra prática:
      PSI < 0.10  -> baixo
      PSI < 0.25  -> moderado
      PSI >= 0.25 -> alto
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.quantile(expected, quantiles)

    # Evita bins duplicados quando a feature tem pouca variação.
    bins = np.unique(bins)

    if len(bins) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def classify_drift(max_abs_z_shift: float, max_psi: float, min_ks_pvalue: float) -> str:
    """
    Classifica o nível geral de drift.

    A regra é conservadora:
      - drift alto se qualquer métrica sinalizar mudança forte
      - drift moderado se houver mudança perceptível
      - drift baixo se tudo estiver estável
    """
    if max_abs_z_shift >= 1.5 or max_psi >= 0.25 or min_ks_pvalue < 0.001:
        return "high"

    if max_abs_z_shift >= 0.75 or max_psi >= 0.10 or min_ks_pvalue < 0.05:
        return "moderate"

    return "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Caminho para o dataset NPZ processado.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split a ser analisado.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Caminho para drift_reference.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho de saída do drift_report.json.",
    )
    args = parser.parse_args()

    with open(args.reference, "r", encoding="utf-8") as f:
        reference = json.load(f)

    ref_stats: dict = reference["reference_stats"]
    names = feature_names()

    # Reconstruct per-feature arrays from the stored summary stats.
    reference_mean = np.asarray([ref_stats[n]["mean"] for n in names], dtype=np.float32)
    reference_std = np.asarray([ref_stats[n]["std"] for n in names], dtype=np.float32)

    # Build a synthetic reference sample from stored percentiles so that PSI
    # and KS tests have something to compare against.  Using interpolation on
    # the five stored quantile points preserves the shape of the reference
    # distribution better than assuming normality.
    _quantile_levels = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    _n_synthetic = 1000
    _interp_levels = np.linspace(0.0, 1.0, _n_synthetic)
    synthetic_cols = []
    for n in names:
        s = ref_stats[n]
        anchors = np.array([s["p05"], s["p25"], s["p50"], s["p75"], s["p95"]])
        col = np.interp(_interp_levels, _quantile_levels, anchors).astype(np.float32)
        synthetic_cols.append(col)
    reference_features = np.column_stack(synthetic_cols)

    sample_rate: float = reference["profile"]["sampling_rate"]

    X_current = load_npz_split(args.dataset, args.split)
    current_features = compute_feature_matrix(X_current, sample_rate=sample_rate)

    current_mean = current_features.mean(axis=0)
    current_std = current_features.std(axis=0)

    feature_reports = []

    for i, name in enumerate(names):
        ref_col = reference_features[:, i]
        cur_col = current_features[:, i]

        z_shift = float((current_mean[i] - reference_mean[i]) / (reference_std[i] + 1e-8))
        feature_psi = psi(ref_col, cur_col)

        ks_stat, ks_pvalue = ks_2samp(ref_col, cur_col)

        feature_reports.append(
            {
                "feature": name,
                "reference_mean": float(reference_mean[i]),
                "current_mean": float(current_mean[i]),
                "reference_std": float(reference_std[i]),
                "current_std": float(current_std[i]),
                "z_shift": z_shift,
                "abs_z_shift": abs(z_shift),
                "psi": float(feature_psi),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
            }
        )

    max_abs_z_shift = max(item["abs_z_shift"] for item in feature_reports)
    max_psi = max(item["psi"] for item in feature_reports)
    min_ks_pvalue = min(item["ks_pvalue"] for item in feature_reports)

    drift_level = classify_drift(
        max_abs_z_shift=max_abs_z_shift,
        max_psi=max_psi,
        min_ks_pvalue=min_ks_pvalue,
    )

    report = {
        "dataset": str(args.dataset),
        "split": args.split,
        "reference_path": str(args.reference),
        "n_reference_samples": int(reference["n_samples"]),
        "n_current_samples": int(len(X_current)),
        "n_features": int(len(names)),
        "summary": {
            "drift_level": drift_level,
            "max_abs_z_shift": float(max_abs_z_shift),
            "max_psi": float(max_psi),
            "min_ks_pvalue": float(min_ks_pvalue),
        },
        "thresholds": {
            "moderate_abs_z_shift": 0.75,
            "high_abs_z_shift": 1.5,
            "moderate_psi": 0.10,
            "high_psi": 0.25,
            "moderate_ks_pvalue": 0.05,
            "high_ks_pvalue": 0.001,
        },
        "features": feature_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("RELATORIO DE DRIFT GERADO")
    print("=" * 80)
    print(f"Dataset:      {args.dataset}")
    print(f"Split:        {args.split}")
    print(f"Amostras:     {len(X_current)}")
    print(f"Drift level:  {drift_level}")
    print(f"Max z-shift:  {max_abs_z_shift:.4f}")
    print(f"Max PSI:      {max_psi:.4f}")
    print(f"Min KS p-val: {min_ks_pvalue:.6f}")
    print(f"Saida:        {args.output}")


if __name__ == "__main__":
    main()