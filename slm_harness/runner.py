"""Experiment runner: executes (condition x task x seed) attempts and logs JSONL.

The runner plays the role of the fixed frontier-model orchestrator deterministically:
it delegates one navigation task to the subagent-under-test and accepts whatever the
harness returns (answer / escalation / failure). Holding the orchestrator fixed and
deterministic removes orchestrator variance from the subagent comparison.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid
from pathlib import Path
from typing import Callable, Iterable

from slm_harness.conditions import default_experiment_config
from slm_harness.harness.custom import CustomSearchHarness
from slm_harness.harness.generic import GenericHarness
from slm_harness.model_clients.base import SubagentModelClient
from slm_harness.model_clients.openai_adapter import OpenAICompatAdapter
from slm_harness.model_clients.stub import OracleStubClient
from slm_harness.schemas.config import ConditionDef, ExperimentConfig, ModelProfile
from slm_harness.schemas.runlog import RunRecord, VerifierResult, append_jsonl
from slm_harness.schemas.task import NavTask, TaskSuite

ClientFactory = Callable[[ModelProfile, int], SubagentModelClient]


def default_client_factory(profile: ModelProfile, seed: int) -> SubagentModelClient:
    """Map a model profile to a client. stub => offline OracleStubClient."""
    if profile.provider == "stub":
        return OracleStubClient(skill=profile.stub_skill, seed=seed)
    if profile.provider == "anthropic":
        raise RuntimeError(
            "the anthropic provider was removed in the standalone release; "
            "use openai_compat (vLLM) or stub"
        )
    if profile.provider == "openai_compat":
        return OpenAICompatAdapter(profile)
    raise ValueError(f"unknown provider {profile.provider!r}")


def build_harness(
    condition: ConditionDef,
    config: ExperimentConfig,
    client: SubagentModelClient,
) -> CustomSearchHarness | GenericHarness:
    """Instantiate the harness side of a condition."""
    model = config.models[condition.model_profile_id]
    harness_profile = config.harnesses[condition.harness_profile_id]
    if condition.harness_factor == "custom":
        return CustomSearchHarness(client, model, harness_profile)
    return GenericHarness(client, model, harness_profile)


async def run_attempt(
    config: ExperimentConfig,
    condition: ConditionDef,
    task: NavTask,
    seed: int,
    client_factory: ClientFactory = default_client_factory,
) -> RunRecord:
    """Run one (condition, task, seed) attempt and return a complete RunRecord."""
    model = config.models[condition.model_profile_id]
    client = client_factory(model, seed)
    harness = build_harness(condition, config, client)
    started = _dt.datetime.now(_dt.timezone.utc)
    outcome = await harness.run(task)
    finished = _dt.datetime.now(_dt.timezone.utc)
    cost = model.pricing.cost_usd(outcome.input_tokens, outcome.output_tokens)
    record = RunRecord(
        run_id=uuid.uuid4().hex[:12],
        experiment_id=config.experiment_id,
        condition_id=condition.condition_id,
        task_id=task.task_id,
        seed=seed,
        model_profile=model.profile_id,
        harness_profile=condition.harness_profile_id,
        success=outcome.success,
        error_type=outcome.error_type,  # type: ignore[arg-type]
        escalated=outcome.escalated,
        final_answer=outcome.final_answer,
        final_verifier=(
            VerifierResult.model_validate(outcome.final_verifier)
            if outcome.final_verifier
            else None
        ),
        steps=outcome.steps,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cost_usd=cost,
        latency_s=outcome.latency_s,
        retries=outcome.retries,
        invalid_actions=outcome.invalid_actions,
        verifier_failures=outcome.verifier_failures,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
    )
    close = getattr(client, "close", None)
    if close is not None:
        await close()
    return record


async def run_experiment(
    config: ExperimentConfig,
    *,
    condition_ids: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
    client_factory: ClientFactory = default_client_factory,
    quiet: bool = False,
) -> list[RunRecord]:
    """Run the full grid and append records to <output_dir>/<runs_filename>."""
    suite = TaskSuite.load(config.tasks_path)
    tasks = suite.tasks
    if splits is not None:
        wanted = set(splits)
        tasks = [t for t in tasks if t.split in wanted]
    conds = [
        config.conditions[cid]
        for cid in (condition_ids or sorted(config.conditions.keys()))
    ]
    out_dir = Path(config.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / config.run.runs_filename
    (out_dir / "config_snapshot.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )

    sem = asyncio.Semaphore(config.run.max_concurrency)
    records: list[RunRecord] = []
    write_lock = asyncio.Lock()

    async def _one(cond: ConditionDef, task: NavTask, seed: int) -> None:
        async with sem:
            record = await run_attempt(config, cond, task, seed, client_factory)
        async with write_lock:
            append_jsonl(runs_path, record)
            records.append(record)
            if not quiet:
                mark = "PASS" if record.success else f"FAIL({record.error_type})"
                print(
                    f"[{record.condition_id}] {record.task_id} seed={seed} {mark} "
                    f"tokens={record.input_tokens}+{record.output_tokens} "
                    f"cost=${record.cost_usd:.6f}"
                )

    jobs = [
        _one(cond, task, seed)
        for cond in conds
        for task in tasks
        for seed in config.run.seeds
    ]
    await asyncio.gather(*jobs)
    return records


def run_smoke(
    *,
    tasks_path: str | None = None,
    seeds: list[int] | None = None,
    output_dir: str = "slm_harness/results/smoke",
    condition_ids: Iterable[str] | None = None,
) -> list[RunRecord]:
    """Synchronous entry point: offline smoke benchmark with stub models."""
    from slm_harness._bootstrap import REPO_ROOT

    tasks = tasks_path or str(REPO_ROOT / "slm_harness/tasks/smoke/smoke_tasks.json")
    config = default_experiment_config(
        tasks, provider="stub", seeds=seeds, output_dir=output_dir
    )
    return asyncio.run(run_experiment(config, condition_ids=condition_ids))
