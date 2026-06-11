"""Compute metrics + attribution decomposition from a runs JSONL file.

Usage:
    python slm_harness/scripts/compute_metrics.py \
        --runs slm_harness/results/smoke/runs.jsonl \
        [--finetune-cost 250.0] [--out metrics.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from slm_harness.metrics import full_report
from slm_harness.schemas.runlog import load_jsonl


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--finetune-cost", type=float, default=0.0)
    parser.add_argument("--out", default=None, help="output JSON path (default: alongside runs)")
    args = parser.parse_args()

    records = load_jsonl(args.runs)
    report = full_report(records, finetune_cost_usd=args.finetune_cost)
    out_path = Path(args.out) if args.out else Path(args.runs).parent / "metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["primary_claim"], indent=2))
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
