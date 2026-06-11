"""Distillation dataset builder.

Converts successful frontier-model trajectories (C1 generic-harness and/or C6
custom-harness) into SFT examples in the custom harness's strict format:

    input  = the minimal JSON observation
    output = the valid next-action JSON

Inclusion rules: only runs with final ``success=True`` are used; within them, only
steps whose action parsed AND passed the step verifier. C6 steps are taken verbatim
(their StepLog already stores observation + parsed action). C1 generic trajectories
are *replayed*: each grep/read tool call is mapped to SEARCH/READ, re-executed
against the (deterministic, read-only) task workspace, and folded through a
``CustomHarnessState`` so the recorded observation is exactly what the custom
harness would have shown — the same schema the fine-tuned SLM will see at
inference time.

Output: JSONL in chat format compatible with common SFT/LoRA pipelines
(TRL SFTTrainer, axolotl, LLaMA-Factory):
    {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}],
     "meta": {...}}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from slm_harness import _bootstrap  # noqa: F401

from openharness.tools.base import ToolExecutionContext

from slm_harness.harness.custom import (
    CUSTOM_SYSTEM_PROMPT,
    execute_env_action,
    fold_result_into_state,
)
from slm_harness.harness.state import CustomHarnessState
from slm_harness.schemas.actions import (
    AnswerAction,
    ReadAction,
    SearchAction,
    SubagentAction,
)
from slm_harness.schemas.runlog import RunRecord
from slm_harness.schemas.task import NavTask, TaskSuite


@dataclass
class DistillExample:
    """One SFT example."""

    observation: str
    action_json: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_chat(self) -> dict[str, Any]:
        """Render in chat-messages SFT format."""
        return {
            "messages": [
                {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
                {"role": "user", "content": self.observation},
                {"role": "assistant", "content": self.action_json},
            ],
            "meta": self.meta,
        }


def _canonical(action: dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def assign_split(task_id: str, task: NavTask | None) -> str:
    """Explicit val/test follow the task; everything else hash-splits 80/10/10.

    The task-declared split is the benchmark train/test split, and every
    distillation source task is declared "train" - honoring it starves the
    SFT val split entirely.
    """
    if task is not None and task.split in ("val", "test"):
        return task.split
    bucket = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 10
    return "train" if bucket < 8 else ("val" if bucket < 9 else "test")


def examples_from_custom_run(record: RunRecord, task: NavTask | None) -> list[DistillExample]:
    """C6-style record: take verified (observation, action) pairs verbatim."""
    examples: list[DistillExample] = []
    for step in record.steps:
        if not step.action_valid or step.parsed_action is None:
            continue
        if step.verifier is not None and not step.verifier.ok:
            continue
        examples.append(
            DistillExample(
                observation=step.observation,
                action_json=_canonical(step.parsed_action),
                meta={
                    "run_id": record.run_id,
                    "condition_id": record.condition_id,
                    "task_id": record.task_id,
                    "step_index": step.step_index,
                    "source": "custom_verbatim",
                    "split": assign_split(record.task_id, task),
                },
            )
        )
    return examples


def _map_generic_step_to_action(step_tool_calls: list[dict[str, Any]]) -> SubagentAction | None:
    """Map a generic-harness tool call onto the custom action set (clean cases only)."""
    if len(step_tool_calls) != 1:
        return None  # parallel/multi-tool steps have no clean single-action equivalent
    call = step_tool_calls[0]
    name, args = call["name"], call.get("input", {})
    if name == "grep" and args.get("pattern"):
        return SearchAction(
            pattern=str(args["pattern"]),
            file_glob=str(args.get("file_glob") or "**/*"),
            root=str(args.get("root") or "."),
        )
    if name == "read_file" and args.get("path"):
        return ReadAction(
            path=str(args["path"]),
            offset=int(args.get("offset") or 0),
            limit=min(400, int(args.get("limit") or 80)),
        )
    return None


async def examples_from_generic_run(
    record: RunRecord, task: NavTask
) -> list[DistillExample]:
    """C1-style record: replay tool calls through the custom-harness state machine."""
    workspace = Path(task.workspace).resolve()
    if not workspace.is_dir():
        return []
    ctx = ToolExecutionContext(cwd=workspace)
    state = CustomHarnessState(goal=task.prompt, max_steps=len(record.steps) + 2)
    examples: list[DistillExample] = []
    split = assign_split(record.task_id, task)

    def _emit(action: SubagentAction, step_index: int, source: str) -> None:
        examples.append(
            DistillExample(
                observation=state.build_observation(),
                action_json=_canonical(action.model_dump()),
                meta={
                    "run_id": record.run_id,
                    "condition_id": record.condition_id,
                    "task_id": record.task_id,
                    "step_index": step_index,
                    "source": source,
                    "split": split,
                },
            )
        )

    for step in record.steps:
        if step.parsed_action and step.parsed_action.get("action") == "FINAL_ANSWER":
            files = [f for f in step.parsed_action.get("files", []) if f]
            if not files or not state.reads:
                return []  # cannot express a valid ANSWER in the custom schema
            first_read = state.reads[0]
            answer = AnswerAction(
                files=files[:5],
                evidence=[
                    {
                        "path": first_read.path,
                        "line_start": first_read.offset + 1,
                        "line_end": first_read.offset + max(1, first_read.n_lines),
                    }
                ],
                answer_text=f"Found in {', '.join(files[:5])}",
                confidence=0.9,
            )
            _emit(answer, step.step_index, "generic_replay")
            return examples
        if not step.tool_calls:
            continue
        action = _map_generic_step_to_action([tc.model_dump() for tc in step.tool_calls])
        if action is None:
            continue  # skip unmappable steps (distractor calls, parallel calls)
        # The emitted training pair must be (state-before, action); only then execute.
        if action.action in ("SEARCH",) or (action.action == "READ" and state.known_files):
            _emit(action, step.step_index, "generic_replay")
        _, output, is_error = await execute_env_action(action, ctx)
        if is_error:
            continue
        fold_result_into_state(action, output, state, workspace)
    return examples


async def build_distillation_dataset(
    records: Iterable[RunRecord],
    suite: TaskSuite,
    *,
    source_conditions: tuple[str, ...] = ("C1", "C6"),
    out_dir: str | Path = "slm_harness/results/distillation",
) -> dict[str, int]:
    """Build train/val/test SFT JSONL from successful, verifier-clean trajectories."""
    tasks_by_id = {t.task_id: t for t in suite.tasks}
    examples: list[DistillExample] = []
    for record in records:
        if record.condition_id not in source_conditions or not record.success:
            continue
        task = tasks_by_id.get(record.task_id)
        if record.harness_profile == "custom":
            examples.extend(examples_from_custom_run(record, task))
        elif task is not None:
            examples.extend(await examples_from_generic_run(record, task))

    # De-duplicate identical (observation, action) pairs.
    seen: set[tuple[str, str]] = set()
    unique: list[DistillExample] = []
    for ex in examples:
        key = (ex.observation, ex.action_json)
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    # With few successful trajectories the hash split can still starve val;
    # SFT needs at least one eval example, so promote ~10% of train.
    splits = [str(ex.meta.get("split", "train")) for ex in unique]
    if "val" not in splits:
        train_idx = [i for i, s in enumerate(splits) if s == "train"]
        if len(train_idx) > 1:
            for i in train_idx[::10]:
                unique[i].meta["split"] = "val"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    handles = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in counts}
    try:
        for ex in unique:
            split = str(ex.meta.get("split", "train"))
            handles[split].write(json.dumps(ex.to_chat(), ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for fh in handles.values():
            fh.close()
    (out / "stats.json").write_text(
        json.dumps(
            {
                "total_examples": len(unique),
                "splits": counts,
                "source_conditions": list(source_conditions),
                "dedup_removed": len(examples) - len(unique),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def build_distillation_dataset_sync(*args: Any, **kwargs: Any) -> dict[str, int]:
    """Synchronous wrapper."""
    return asyncio.run(build_distillation_dataset(*args, **kwargs))
