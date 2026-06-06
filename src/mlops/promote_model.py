"""
Quality gate para promocao de modelo.

Este script decide se o modelo escolhido pelo treinamento pode virar
"modelo de producao".

Entrada:
  artefacts/reports/candidate_manifest.json

Saidas:
  artefacts/reports/promotion_report.json
  artefacts/registry/production_manifest.json  (somente se aprovado)

Ideia:
  Treinar bem nao basta.
  O modelo precisa passar por regras minimas antes de ser candidato a OTA.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.settings import ARTEFACTS_DIR, QUALITY_GATE, REPORTS_DIR


REGISTRY_DIR = ARTEFACTS_DIR / "registry"
CANDIDATE_MANIFEST = REPORTS_DIR / "candidate_manifest.json"
PROMOTION_REPORT = REPORTS_DIR / "promotion_report.json"
PRODUCTION_MANIFEST = REGISTRY_DIR / "production_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    """Carrega um JSON e retorna um dicionario."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Salva um dicionario em JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    """
    Busca segura em dicionarios aninhados.

    Exemplo:
      get_nested(candidate, ["summary_metrics", "test_auc_pr"])
    """
    current: Any = payload

    for key in path:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def model_size_kb(model_path: str | None) -> float | None:
    """Calcula tamanho do arquivo do modelo em KB."""
    if not model_path:
        return None

    path = Path(model_path)
    if not path.exists():
        return None

    return path.stat().st_size / 1024.0


def evaluate_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Aplica as regras do quality gate.

    Retorna um relatorio contendo:
      - approved: bool
      - checks: lista de verificacoes
      - reasons: motivos de reprovacao
    """
    summary = candidate.get("summary_metrics", {})

    test_auc_pr = summary.get("test_auc_pr")
    test_f1 = summary.get("test_f1")
    test_fp_per_hour = summary.get("test_fp_per_hour")
    val_auc_pr = summary.get("val_auc_pr")

    model_path = candidate.get("model_path")
    size_kb = model_size_kb(model_path)

    min_auc_pr = float(QUALITY_GATE.get("min_auc_pr", 0.0))
    min_f1 = float(QUALITY_GATE.get("min_f1", 0.0))
    max_model_size_kb = QUALITY_GATE.get("max_model_size_kb")
    max_fp_per_hour = QUALITY_GATE.get("max_fp_per_hour")
    max_val_test_auc_pr_gap = QUALITY_GATE.get("max_val_test_auc_pr_gap")

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    def add_check(
        name: str,
        passed: bool,
        value: Any,
        rule: str,
    ) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "value": value,
                "rule": rule,
            }
        )
        if not passed:
            reasons.append(f"{name} falhou: valor={value}, regra={rule}")

    add_check(
        name="min_auc_pr",
        passed=test_auc_pr is not None and float(test_auc_pr) >= min_auc_pr,
        value=test_auc_pr,
        rule=f">= {min_auc_pr}",
    )

    add_check(
        name="min_f1",
        passed=test_f1 is not None and float(test_f1) >= min_f1,
        value=test_f1,
        rule=f">= {min_f1}",
    )

    if max_fp_per_hour is not None:
        max_fp_per_hour = float(max_fp_per_hour)
        add_check(
            name="max_fp_per_hour",
            passed=(
                test_fp_per_hour is not None
                and float(test_fp_per_hour) <= max_fp_per_hour
            ),
            value=test_fp_per_hour,
            rule=f"<= {max_fp_per_hour}",
        )

    if max_val_test_auc_pr_gap is not None:
        max_val_test_auc_pr_gap = float(max_val_test_auc_pr_gap)

        if val_auc_pr is None or test_auc_pr is None:
            gap = None
            passed = False
        else:
            gap = abs(float(val_auc_pr) - float(test_auc_pr))
            passed = gap <= max_val_test_auc_pr_gap

        add_check(
            name="max_val_test_auc_pr_gap",
            passed=passed,
            value=gap,
            rule=f"<= {max_val_test_auc_pr_gap}",
        )

    if max_model_size_kb is not None:
        max_model_size_kb = float(max_model_size_kb)
        add_check(
            name="max_model_size_kb",
            passed=size_kb is not None and size_kb <= max_model_size_kb,
            value=size_kb,
            rule=f"<= {max_model_size_kb}",
        )

    approved = all(check["passed"] for check in checks)

    return {
        "approved": approved,
        "candidate_model": candidate.get("model_name"),
        "candidate_family": candidate.get("family"),
        "model_path": model_path,
        "model_size_kb": size_kb,
        "checks": checks,
        "reasons": reasons,
    }


def build_production_manifest(
    candidate: dict[str, Any],
    gate_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria manifesto de producao.

    Esse arquivo vira a fonte de verdade para a etapa futura de OTA.
    """
    return {
        "status": "production",
        "model_name": candidate.get("model_name"),
        "family": candidate.get("family"),
        "model_path": candidate.get("model_path"),
        "threshold": candidate.get("threshold"),
        "profile": candidate.get("profile"),
        "dataset": candidate.get("dataset"),
        "summary_metrics": candidate.get("summary_metrics"),
        "params": candidate.get("params"),
        "parameter_count": candidate.get("parameter_count"),
        "quality_gate": gate_report,
    }


def main() -> None:
    candidate = load_json(CANDIDATE_MANIFEST)

    gate_report = evaluate_gate(candidate)

    save_json(PROMOTION_REPORT, gate_report)

    if gate_report["approved"]:
        production_manifest = build_production_manifest(
            candidate=candidate,
            gate_report=gate_report,
        )
        save_json(PRODUCTION_MANIFEST, production_manifest)

        print("=" * 80)
        print("MODELO APROVADO NO QUALITY GATE")
        print("=" * 80)
        print(f"Modelo: {candidate.get('model_name')}")
        print(f"Manifesto de producao: {PRODUCTION_MANIFEST}")
    else:
        print("=" * 80)
        print("MODELO REPROVADO NO QUALITY GATE")
        print("=" * 80)
        print(f"Modelo: {candidate.get('model_name')}")
        print("Motivos:")
        for reason in gate_report["reasons"]:
            print(f"  - {reason}")
        print(f"Relatorio: {PROMOTION_REPORT}")


if __name__ == "__main__":
    main()