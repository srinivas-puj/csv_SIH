"""M1 data-integrity entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CV_ASSURE = Path(__file__).resolve().parents[2]
for _path in (str(CV_ASSURE), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pipeline import run_data_integrity  # noqa: E402


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CV-Assure Module M1 data-integrity checks.")
    parser.add_argument("--records", required=True, type=Path, help="JSON list of AssetRecord-like dicts.")
    parser.add_argument("--matrix", required=True, type=Path, help="Capability matrix JSON.")
    parser.add_argument("--embeddings", type=Path, default=None, help="Optional {asset_id: {ref_emb, sub_emb}} JSON.")
    parser.add_argument("--reference-profile", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("m1_result.json"))
    parser.add_argument("--baseline-rate", type=float, default=0.05)
    args = parser.parse_args(argv)

    records = _load_json(args.records)
    matrix = _load_json(args.matrix)
    embeddings = _load_json(args.embeddings) if args.embeddings else None
    profile = _load_json(args.reference_profile) if args.reference_profile else None
    result = run_data_integrity(
        records,
        matrix,
        embeddings=embeddings,
        reference_profile=profile,
        pipeline_baseline_rate=args.baseline_rate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_json(), encoding="utf-8")
    print(args.out)
    return 0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
