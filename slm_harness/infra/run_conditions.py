"""Stage: evaluate conditions on a split against the self-hosted endpoints.

Used for the final C1-C6 evaluation on the held-out TEST split, and reusable for any
subset of conditions. Writes runs JSONL + a metrics report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from slm_harness import _bootstrap  # noqa: F401

from slm_harness.infra.real_experiment import build_real_config
from slm_harness.metrics import full_report
from slm_harness.runner import run_experiment


async def _run(args: argparse.Namespace) -> None:
    cfg = build_real_config(
        experiment_id=args.experiment_id,
        tasks_path=args.tasks,
        teacher_url=args.teacher_url,
        teacher_model=args.teacher_model,
        student_url=args.student_url,
        student_model=args.student_model,
        adapter_name=args.adapter_name,
        output_dir=args.out,
        seeds=list(range(args.seeds)),
        teacher_pricing_json=args.teacher_pricing,
        student_base_pricing_json=args.student_base_pricing,
        student_ft_pricing_json=args.student_ft_pricing,
        finetune_cost_usd=args.finetune_cost,
    )
    conds = args.conditions or ["C1", "C2", "C3", "C4", "C5", "C6"]
    records = await run_experiment(
        cfg, condition_ids=conds, splits=[args.split], quiet=False
    )
    report = full_report(records, finetune_cost_usd=args.finetune_cost)
    out = Path(args.out)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== primary claim (C5 vs C1) ===")
    print(json.dumps(report["primary_claim"], indent=2))


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", default="real")
    p.add_argument("--tasks", required=True, help="test/val task suite JSON")
    p.add_argument("--split", default="test")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--teacher-url", required=True)
    p.add_argument("--teacher-model", default="teacher")
    p.add_argument("--student-url", default="http://localhost:8002/v1")
    p.add_argument("--student-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter-name", default="navsearch-4b")
    p.add_argument("--teacher-pricing", default=None)
    p.add_argument("--student-base-pricing", default=None)
    p.add_argument("--student-ft-pricing", default=None)
    p.add_argument("--finetune-cost", type=float, default=0.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="slm_harness/results/real/eval")
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
