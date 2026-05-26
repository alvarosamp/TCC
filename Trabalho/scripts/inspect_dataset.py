from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcc_pipeline import PipelineProfile
from tcc_pipeline.dataset import inspect_npz_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspeciona um dataset NPZ contra um profile versionado."
    )
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles" / "seismic_v1.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profile = PipelineProfile.from_json(args.profile)
    report = inspect_npz_dataset(args.dataset, profile)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nOK: relatorio salvo em {args.output}")


if __name__ == "__main__":
    main()
