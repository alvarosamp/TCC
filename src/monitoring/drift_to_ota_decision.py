"""
Etapa 4 - Conexão entre drift, retreino e OTA.

Este script não treina nem publica OTA diretamente.
Ele decide se o sistema está pronto para acionar o fluxo OTA.

Entrada:
  - artefacts/monitoring/retrain_policy.json
  - artefacts/registry/production_manifest.json
  - artefacts/reports/candidate_manifest.json opcional

Saída:
  - artefacts/monitoring/ota_decision.json

Lógica:
  - Se não precisa retreinar: não aciona OTA.
  - Se precisa retreinar, mas não existe candidato: aguarda treinamento.
  - Se o candidato ja foi promovido para producao: recomenda gerar OTA.
  - Se candidato não aprovado: mantém modelo atual.
"""

import argparse
import json
from pathlib import Path


DEFAULT_POLICY_PATH = Path("artefacts/monitoring/retrain_policy.json")
DEFAULT_PRODUCTION_MANIFEST_PATH = Path("artefacts/registry/production_manifest.json")
DEFAULT_CANDIDATE_MANIFEST_PATH = Path("artefacts/reports/candidate_manifest.json")
DEFAULT_OUTPUT_PATH = Path("artefacts/monitoring/ota_decision.json")


def read_json_if_exists(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_value(manifest: dict | None, keys: tuple[str, ...]):
    """Retorna o primeiro valor existente em uma lista de chaves."""
    if manifest is None:
        return None

    for key in keys:
        value = manifest.get(key)
        if value is not None:
            return value

    return None


def same_candidate_as_production(
    candidate_manifest: dict | None,
    production_manifest: dict | None,
) -> bool:
    """
    Confere se o modelo candidato e o modelo promovido sao o mesmo artefato logico.

    O train_all gera candidate_manifest em artefacts/reports.
    O promote_model, quando aprova, copia esse candidato para
    production_manifest em artefacts/registry.
    """
    if candidate_manifest is None or production_manifest is None:
        return False

    candidate_model = manifest_value(candidate_manifest, ("model_name", "model"))
    production_model = manifest_value(production_manifest, ("model_name", "model"))

    if candidate_model is None or candidate_model != production_model:
        return False

    candidate_path = manifest_value(candidate_manifest, ("model_path", "artifact_path"))
    production_path = manifest_value(production_manifest, ("model_path", "artifact_path"))

    if candidate_path is not None and production_path is not None:
        return str(candidate_path) == str(production_path)

    candidate_metrics = candidate_manifest.get("summary_metrics")
    production_metrics = production_manifest.get("summary_metrics")
    return candidate_metrics == production_metrics


def production_gate_is_approved(production_manifest: dict | None) -> bool:
    """Verifica se o manifesto de producao foi criado por um quality gate aprovado."""
    if production_manifest is None:
        return False

    if production_manifest.get("status") == "production":
        quality_gate = production_manifest.get("quality_gate", {})
        if isinstance(quality_gate, dict):
            return quality_gate.get("approved") is True

    return False


def candidate_is_approved(
    candidate_manifest: dict | None,
    production_manifest: dict | None = None,
) -> bool:
    """
    Verifica se existe candidato aprovado.

    A aprovacao pode vir direto do candidato ou indiretamente do
    production_manifest criado pelo promote_model.
    """
    if candidate_manifest is None:
        return False

    if candidate_manifest.get("approved") is True:
        return True

    if candidate_manifest.get("quality_gate_passed") is True:
        return True

    if candidate_manifest.get("status") in {"approved", "promoted", "passed", "production"}:
        return True

    quality_gate = candidate_manifest.get("quality_gate", {})
    if isinstance(quality_gate, dict) and (
        quality_gate.get("passed") is True or quality_gate.get("approved") is True
    ):
        return True

    if (
        production_gate_is_approved(production_manifest)
        and same_candidate_as_production(candidate_manifest, production_manifest)
    ):
        return True

    return False

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Caminho para retrain_policy.json.",
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=DEFAULT_PRODUCTION_MANIFEST_PATH,
        help="Manifesto do modelo atual em producao.",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=DEFAULT_CANDIDATE_MANIFEST_PATH,
        help="Manifesto do novo modelo candidato.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho de saída para ota_decision.json.",
    )
    args = parser.parse_args()

    policy = read_json_if_exists(args.policy)
    production_manifest = read_json_if_exists(args.production_manifest)
    candidate_manifest = read_json_if_exists(args.candidate_manifest)

    if policy is None:
        raise FileNotFoundError(f"Politica de retreino nao encontrada: {args.policy}")

    decision = policy["decision"]
    should_retrain = bool(decision["should_retrain"])

    production_version = None
    if production_manifest is not None:
        production_version = (
            production_manifest.get("version")
            or production_manifest.get("model_version")
            or production_manifest.get("run_id")
        )

    candidate_version = None
    if candidate_manifest is not None:
        candidate_version = (
            candidate_manifest.get("version")
            or candidate_manifest.get("model_version")
            or candidate_manifest.get("run_id")
        )

    approved_candidate = candidate_is_approved(candidate_manifest, production_manifest)

    if not should_retrain:
        ota_action = "do_not_publish_ota"
        reason = "Politica de drift nao recomenda retreino."

    elif candidate_manifest is None:
        ota_action = "wait_for_new_candidate"
        reason = "Drift recomenda retreino, mas ainda nao existe modelo candidato."

    elif not approved_candidate:
        ota_action = "keep_current_model"
        reason = "Existe candidato, mas ele ainda nao foi aprovado no quality gate."

    else:
        ota_action = "build_and_publish_ota"
        reason = "Drift recomendou retreino e existe candidato aprovado."

    ota_decision = {
        "policy_path": str(args.policy),
        "production_manifest_path": str(args.production_manifest),
        "candidate_manifest_path": str(args.candidate_manifest),
        "production_version": production_version,
        "candidate_version": candidate_version,
        "should_retrain": should_retrain,
        "candidate_available": candidate_manifest is not None,
        "candidate_approved": approved_candidate,
        "ota_action": ota_action,
        "reason": reason,
        "suggested_commands": [],
    }

    if ota_action == "build_and_publish_ota":
        ota_decision["suggested_commands"] = [
            "python -m src.mlops.promote_model",
            "python -m src.export.export_tflite",
            "python -m src.ota.build_ota_manifest",
            "python -m src.ota.build_ota_package",
            "python -m src.ota.validate_ota_package",
            "python -m src.ota.publish_local_release",
        ]

    elif ota_action == "wait_for_new_candidate":
        ota_decision["suggested_commands"] = [
            "python -m src.training.train_all --models-cfg config/model/models.yaml",
            "python -m src.mlops.promote_model",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ota_decision, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("DECISAO DRIFT -> OTA GERADA")
    print("=" * 80)
    print(f"Acao OTA:     {ota_action}")
    print(f"Retreinar:    {should_retrain}")
    print(f"Candidato:    {candidate_manifest is not None}")
    print(f"Aprovado:     {approved_candidate}")
    print(f"Motivo:       {reason}")
    print(f"Saida:        {args.output}")

    if ota_decision["suggested_commands"]:
        print()
        print("Comandos sugeridos:")
        for command in ota_decision["suggested_commands"]:
            print(f"  {command}")


if __name__ == "__main__":
    main()