from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc_seismic.paths import results_dir


def collect_results(base: Path) -> list[dict]:
    rows = []
    for path in base.glob("**/results.json"):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        test = payload["evaluation"]["test"]
        rows.append(
            {
                "model": payload["model"],
                "split": payload["split"],
                "auc_pr": test["auc_pr"],
                "auc_roc": test["auc_roc"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "threshold": test["threshold"],
                "params_total": payload.get("params_total"),
            }
        )

    classical = base / "classical_ml" / "summary.json"
    if classical.exists():
        with classical.open(encoding="utf-8") as f:
            payload = json.load(f)
        for result in payload["results"]:
            if result.get("skipped"):
                continue
            test = result["evaluation"]["test"]
            rows.append(
                {
                    "model": result["model"],
                    "split": result["split"],
                    "auc_pr": test["auc_pr"],
                    "auc_roc": test["auc_roc"],
                    "precision": test["precision"],
                    "recall": test["recall"],
                    "f1": test["f1"],
                    "threshold": test["threshold"],
                    "params_total": None,
                }
            )
    return rows


def main() -> None:
    base = results_dir("models_corrected")
    rows = collect_results(base)
    if not rows:
        print(f"No corrected results found in {base}")
        return

    rows.sort(key=lambda r: (r["split"], -(r["auc_pr"] or -1), r["model"]))
    csv_path = base / "comparison_corrected.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = [
            "split",
            "model",
            "auc_pr",
            "auc_roc",
            "precision",
            "recall",
            "f1",
            "threshold",
            "params_total",
        ]
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if row[h] is None else str(row[h]) for h in header) + "\n")

    print(f"Saved: {csv_path}")
    print(f"{'split':<14} {'model':<22} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8}")
    for row in rows:
        print(
            f"{row['split']:<14} {row['model']:<22} "
            f"{row['auc_pr'] if row['auc_pr'] is not None else 0:>8.4f} "
            f"{row['auc_roc'] if row['auc_roc'] is not None else 0:>8.4f} "
            f"{row['f1']:>8.4f}"
        )


if __name__ == "__main__":
    main()

