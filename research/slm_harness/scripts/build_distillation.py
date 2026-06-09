"""Build the SFT distillation dataset from successful C1/C6 trajectories.

Usage:
    python research/slm_harness/scripts/build_distillation.py \
        --runs research/slm_harness/results/smoke/runs.jsonl \
        --tasks research/slm_harness/tasks/smoke/smoke_tasks.json \
        [--sources C1 C6] [--out research/slm_harness/results/distillation]
"""

from __future__ import annotations

import argparse

import _path  # noqa: F401

from research.slm_harness.distill import build_distillation_dataset_sync
from research.slm_harness.schemas.runlog import load_jsonl
from research.slm_harness.schemas.task import TaskSuite


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--sources", nargs="*", default=["C1", "C6"])
    parser.add_argument("--out", default="research/slm_harness/results/distillation")
    args = parser.parse_args()

    records = load_jsonl(args.runs)
    suite = TaskSuite.load(args.tasks)
    counts = build_distillation_dataset_sync(
        records, suite, source_conditions=tuple(args.sources), out_dir=args.out
    )
    print(f"Wrote SFT JSONL to {args.out}: {counts}")


if __name__ == "__main__":
    main()
