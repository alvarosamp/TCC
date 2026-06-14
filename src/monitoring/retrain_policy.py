"""
Etapa 3 - Política de retreinamento.

Lê o relatório de drift e decide a próxima ação do sistema.

Entrada:
  - artefacts/monitoring/drift_report.json

Saída:
  - artefacts/monitoring/retrain_policy.json

Ações possíveis:
  - keep_current_model
  - monitor_more_frequently
  - retrain_recommended
  - retrain_required
"""

import argparse
import json
from pathlib import Path


DEFAULT_DRIFT_REPORT_PATH = Path("artefacts/monitoring/drift_report.json")
DEFAULT_OUTPUT_PATH = Path("artefacts/monitoring/retrain_policy.json")


def decide_action(drift_level: str, max_psi: float, max_abs_z_shift: float) -> dict:
    """
    Decide a ação com base no nível de drift.

    A ideia é separar:
      - drift baixo: nenhuma ação
      - drift moderado: acompanhar
      - drift alto: retreinar
      - drift muito alto: retreino obrigatório
    """
    if drift_level == "low":
        return {
            "action": "keep_current_model",
            "should_retrain": False,
            "should_trigger_ota": False,
            "reason": "Distribuicao atual semelhante a referencia de treino.",
        }

    if drift_level == "moderate":
        return {
            "action": "monitor_more_frequently",
            "should_retrain": False,
            "should_trigger_ota": False,
            "reason": "Drift moderado detectado. Ainda nao justifica retreino automatico.",
        }

    if max_psi >= 0.40 or max_abs_z_shift >= 2.5:
        return {
            "action": "retrain_required",
            "should_retrain": True,
            "should_trigger_ota": False,
            "reason": "Drift alto severo. Recomenda-se retreino antes de nova atualizacao OTA.",
        }

    return {
        "action": "retrain_recommended",
        "should_retrain": True,
        "should_trigger_ota": False,
        "reason": "Drift alto detectado. Recomenda-se treinar novo candidato e validar no quality gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drift-report",
        type=Path,
        default=DEFAULT_DRIFT_REPORT_PATH,
        help="Caminho para drift_report.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho de saída para retrain_policy.json.",
    )
    args = parser.parse_args()

    with open(args.drift_report, "r", encoding="utf-8") as f:
        drift_report = json.load(f)

    summary = drift_report["summary"]

    drift_level = summary["drift_level"]
    max_psi = float(summary["max_psi"])
    max_abs_z_shift = float(summary["max_abs_z_shift"])
    min_ks_pvalue = float(summary["min_ks_pvalue"])

    decision = decide_action(
        drift_level=drift_level,
        max_psi=max_psi,
        max_abs_z_shift=max_abs_z_shift,
    )

    policy = {
        "drift_report": str(args.drift_report),
        "drift_level": drift_level,
        "metrics": {
            "max_psi": max_psi,
            "max_abs_z_shift": max_abs_z_shift,
            "min_ks_pvalue": min_ks_pvalue,
        },
        "decision": decision,
        "next_steps": [],
    }

    if decision["action"] == "keep_current_model":
        policy["next_steps"] = [
            "Manter modelo atual em producao.",
            "Continuar monitoramento periodico.",
        ]

    elif decision["action"] == "monitor_more_frequently":
        policy["next_steps"] = [
            "Aumentar frequencia de monitoramento.",
            "Coletar novo lote de dados.",
            "Comparar novamente antes de retreinar.",
        ]

    elif decision["action"] in {"retrain_recommended", "retrain_required"}:
        policy["next_steps"] = [
            "Executar novo treinamento com dados atualizados.",
            "Comparar candidato com modelo em producao.",
            "Aplicar quality gate.",
            "Gerar pacote OTA apenas se o novo modelo for aprovado.",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("POLITICA DE RETREINO GERADA")
    print("=" * 80)
    print(f"Drift level: {drift_level}")
    print(f"Acao:        {decision['action']}")
    print(f"Retreinar:   {decision['should_retrain']}")
    print(f"OTA agora:   {decision['should_trigger_ota']}")
    print(f"Motivo:      {decision['reason']}")
    print(f"Saida:       {args.output}")


if __name__ == "__main__":
    main()