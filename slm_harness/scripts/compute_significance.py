"""Compute deduped statistics, CIs, and paired tests from real run logs.

Usage:
    python slm_harness/scripts/compute_significance.py \
        --runs slm_harness/results/real/eval_teacher/runs.jsonl \
               slm_harness/results/real/eval_a_base/runs.jsonl \
               slm_harness/results/real/eval_a_ft/runs.jsonl \
        [--out slm_harness/results/real/significance.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from slm_harness.analysis import significance_report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    report = significance_report(
        args.runs, n_resamples=args.resamples, rng_seed=args.rng_seed
    )

    out = Path(args.out) if args.out else Path(args.runs[0]).parent / "significance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    print("\n| Condition | n | Success (95% CI) | Cost/success | replays id/div |")
    print("|---|---|---|---|---|")
    for cond, s in report["conditions"].items():
        lo, hi = s["success_ci95"]
        cps = "-" if s["cost_per_success"] is None else f"${s['cost_per_success']:.6f}"
        print(
            f"| {cond} | {s['n_tasks']} | {s['success_rate']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] | {cps} | "
            f"{s['replays_identical']}/{s['replays_divergent']} |"
        )
    print("\n| Pair | discordant (A-only/B-only) | McNemar p |")
    print("|---|---|---|")
    for p in report["paired_success"]:
        print(
            f"| {p['condition_a']} vs {p['condition_b']} | "
            f"{p['a_only_success']}/{p['b_only_success']} | {p['mcnemar_p']:.2e} |"
        )
    def fmt(x: float | None, suffix: str = "") -> str:
        return "-" if x is None else f"{x:.3f}{suffix}"

    print("\n| Pair | cost/success ratio [95% CI] | reduction % [95% CI] |")
    print("|---|---|---|")
    for b in report["cost_ratio_bootstrap"]:
        lo, hi = b["ci95"]
        rlo, rhi = b["reduction_pct_ci95"]
        red = b["reduction_pct_point"]
        print(
            f"| {b['condition_a']}/{b['condition_b']} | "
            f"{fmt(b['point_ratio'])} [{fmt(lo)}, {fmt(hi)}] | "
            f"{'-' if red is None else f'{red:.1f}%'} "
            f"[{'-' if rlo is None else f'{rlo:.1f}%'}, "
            f"{'-' if rhi is None else f'{rhi:.1f}%'}] |"
        )


if __name__ == "__main__":
    main()
