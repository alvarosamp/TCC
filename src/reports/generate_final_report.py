"""Gera relatorio final em Markdown e HTML a partir dos manifests do projeto."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

OUT_DIR = Path("docs/results/final_report")
NL = chr(10)

SOURCES = {
    "candidate": Path("artefacts/reports/candidate_manifest.json"),
    "comparison": Path("artefacts/reports/model_comparison.json"),
    "promotion": Path("artefacts/reports/promotion_report.json"),
    "production": Path("artefacts/registry/production_manifest.json"),
    "drift": Path("artefacts/monitoring/drift_report.json"),
    "retrain_policy": Path("artefacts/monitoring/retrain_policy.json"),
    "ota_decision": Path("artefacts/monitoring/ota_decision.json"),
    "prediction_drift": Path("artefacts/monitoring/prediction_performance_drift_report.json"),
    "edge_export": Path("artefacts/edge/tiny_cnn_export_manifest.json"),
    "ota_manifest": Path("artefacts/ota/ota_manifest.json"),
    "latest_release": Path("artefacts/ota/releases/latest.json"),
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def bullet_json(title: str, payload: dict[str, Any] | None) -> str:
    if payload is None:
        return NL.join([f"## {title}", "", "Nao encontrado."])
    json_block = json.dumps(payload, indent=2, ensure_ascii=False)[:6000]
    return NL.join([f"## {title}", "", "```json", json_block, "```"])


def build_markdown(data: dict[str, dict[str, Any] | None]) -> str:
    candidate = data["candidate"] or {}
    metrics = candidate.get("summary_metrics", {})
    drift_summary = (data["drift"] or {}).get("summary", {})
    ota_decision = data["ota_decision"] or {}
    edge = data["edge_export"] or {}
    exports = edge.get("exports", {})

    lines = [
        "# Relatorio Final - Pipeline TinyML/MLOps",
        "",
        "## Resumo Executivo",
        "",
        "Este relatorio consolida treino, quality gate, drift, exportacao edge, OTA e observabilidade.",
        "",
        "## Modelo Selecionado",
        "",
        f"- Modelo: {fmt(candidate.get('model_name'))}",
        f"- AUC-PR teste: {fmt(metrics.get('test_auc_pr'))}",
        f"- F1 teste: {fmt(metrics.get('test_f1'))}",
        f"- Precision: {fmt(metrics.get('test_precision'))}",
        f"- Recall: {fmt(metrics.get('test_recall'))}",
        f"- FP/h: {fmt(metrics.get('test_fp_per_hour'))}",
        f"- Threshold: {fmt(candidate.get('threshold'))}",
        "",
        "## Drift",
        "",
        f"- Nivel: {fmt(drift_summary.get('drift_level'))}",
        f"- Max PSI: {fmt(drift_summary.get('max_psi'))}",
        f"- Max z-shift: {fmt(drift_summary.get('max_abs_z_shift'))}",
        f"- Min KS p-value: {fmt(drift_summary.get('min_ks_pvalue'))}",
        "",
        "## Decisao OTA",
        "",
        f"- Acao: {fmt(ota_decision.get('ota_action'))}",
        f"- Candidato aprovado: {fmt(ota_decision.get('candidate_approved'))}",
        f"- Motivo: {fmt(ota_decision.get('reason'))}",
        "",
        "## Exportacao Edge",
        "",
    ]

    for name, info in exports.items():
        lines.append(f"- {name}: {fmt(info.get('size_kb'))} KB - `{info.get('path')}`")

    lines.extend([
        "",
        "## Manifests Detalhados",
        "",
        bullet_json("Candidate Manifest", data["candidate"]),
        bullet_json("Promotion Report", data["promotion"]),
        bullet_json("Drift Report", data["drift"]),
        bullet_json("Prediction/Performance Drift", data["prediction_drift"]),
        bullet_json("OTA Manifest", data["ota_manifest"]),
    ])
    return NL.join(lines)


def markdown_to_html(markdown: str) -> str:
    body = []
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            body.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
        elif in_code:
            body.append(html.escape(line))
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p class='bullet'>{html.escape(line)}</p>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
        else:
            body.append("")
    head = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatorio Final TinyML/MLOps</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.45; color: #1f2937; }
h1, h2 { color: #111827; }
pre { background: #f3f4f6; padding: 16px; overflow-x: auto; border-radius: 6px; }
.bullet { margin-left: 16px; }
</style>
</head>
<body>
"""
    return head + NL.join(body) + NL + "</body>" + NL + "</html>" + NL


def main() -> None:
    data = {name: load_json(path) for name, path in SOURCES.items()}
    markdown = build_markdown(data)
    html_doc = markdown_to_html(markdown)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "relatorio_final.md"
    html_path = OUT_DIR / "relatorio_final.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")
    print("=" * 80)
    print("RELATORIO FINAL GERADO")
    print("=" * 80)
    print(f"Markdown: {md_path}")
    print(f"HTML:     {html_path}")
    print("PDF:      abra o HTML no navegador e salve/imprima em PDF")


if __name__ == "__main__":
    main()
