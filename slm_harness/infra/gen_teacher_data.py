"""Stage: generate teacher trajectories and build the distillation dataset.

Runs C1 (large + generic) and C6 (large + custom) over the TRAIN split against the
self-hosted teacher endpoint, logs full trajectories, then converts the successful,
verifier-clean ones into SFT JSONL. Zero paid API — the teacher is local vLLM.
"""

from __future__ import annotations

import argparse
import asyncio

from slm_harness import _bootstrap  # noqa: F401

from slm_harness.distill import build_distillation_dataset
from slm_harness.infra.real_experiment import build_real_config
from slm_harness.runner import run_experiment
from slm_harness.schemas.task import TaskSuite


async def _run(args: argparse.Namespace) -> None:
    cfg = build_real_config(
        experiment_id=f"{args.experiment_id}-teacher",
        tasks_path=args.tasks,
        teacher_url=args.teacher_url,
        teacher_model=args.teacher_model,
        student_url=args.student_url,
        student_model=args.student_model,
        adapter_name=args.adapter_name,
        output_dir=args.out,
        seeds=list(range(args.seeds)),
        teacher_pricing_json=args.teacher_pricing,
    )
    # Teacher data-gen uses C1 (generic) + C6 (custom) on the train split.
    records = await run_experiment(
        cfg, condition_ids=["C1", "C6"], splits=["train"], quiet=False
    )
    suite = TaskSuite.load(args.tasks)
    counts = await build_distillation_dataset(
        records,
        suite,
        source_conditions=("C1", "C6"),
        out_dir=args.distill_out,
    )
    n_ok = sum(1 for r in records if r.success)
    print(f"\nteacher trajectories: {len(records)} runs, {n_ok} successful")
    print(f"distillation examples: {counts}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", default="real")
    p.add_argument("--tasks", required=True, help="train task suite JSON")
    p.add_argument("--teacher-url", required=True)
    p.add_argument("--teacher-model", default="teacher")
    p.add_argument("--student-url", default="http://localhost:8002/v1")
    p.add_argument("--student-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter-name", default="navsearch-4b")
    p.add_argument("--teacher-pricing", default=None)
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..n-1)")
    p.add_argument("--out", default="slm_harness/results/real/teacher")
    p.add_argument("--distill-out", default="slm_harness/results/real/distillation")
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
