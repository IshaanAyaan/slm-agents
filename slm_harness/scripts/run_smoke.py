"""Run the offline smoke benchmark (stub models, no network/credentials).

Usage:
    python slm_harness/scripts/run_smoke.py [--conditions C1 C5] [--seeds 0 1 2]
        [--out slm_harness/results/smoke]
"""

from __future__ import annotations

import argparse

import _path  # noqa: F401

from slm_harness.metrics import full_report
from slm_harness.runner import run_smoke


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="*", default=None, help="e.g. C1 C2 C5")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="slm_harness/results/smoke")
    parser.add_argument("--tasks", default=None, help="path to a task suite JSON")
    args = parser.parse_args()

    records = run_smoke(
        tasks_path=args.tasks,
        seeds=args.seeds,
        output_dir=args.out,
        condition_ids=args.conditions,
    )
    report = full_report(records)
    print("\n=== smoke summary (SYNTHETIC stub-model numbers) ===")
    for cid, m in report["conditions"].items():
        cps = m["cost_per_success"]
        print(
            f"{cid}: success={m['success_rate']:.2f} "
            f"cost/success={'inf' if cps is None else f'${cps:.5f}'} "
            f"retries/attempt={m['retry_rate']:.2f}"
        )
    print(f"\nRuns JSONL written under: {args.out}")


if __name__ == "__main__":
    main()
