"""Generate all figures and numbered tables from a runs JSONL file.

Usage:
    python slm_harness/scripts/make_figures.py \
        --runs slm_harness/results/smoke/runs.jsonl \
        [--out slm_harness/results/smoke/figures] [--real]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from slm_harness.figures import generate_all
from slm_harness.schemas.runlog import load_jsonl


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--finetune-cost", type=float, default=250.0)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Set ONLY for real-model runs; removes the synthetic-data watermark.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.runs)
    out_dir = Path(args.out) if args.out else Path(args.runs).parent / "figures"
    written = generate_all(
        records, out_dir, synthetic=not args.real, finetune_cost_usd=args.finetune_cost
    )
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
