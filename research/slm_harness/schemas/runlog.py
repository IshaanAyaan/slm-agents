"""Structured run logging schema (JSONL; one RunRecord per task attempt).

This schema doubles as the distillation source format: each StepLog stores the exact
observation shown to the model and the raw/parsed action, so successful trajectories
can be converted to SFT examples without re-running anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ErrorType = Literal[
    "",
    "invalid_action",
    "verifier_failure",
    "wrong_answer",
    "escalated",
    "budget_exhausted",
    "tool_error",
    "model_error",
]


class ToolCallLog(BaseModel):
    """One environment tool execution inside a step."""

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output_chars: int = 0
    is_error: bool = False


class VerifierResult(BaseModel):
    """Outcome of deterministic verification for one step or the final answer."""

    ok: bool
    reason: str = ""
    retryable: bool = False
    checks: dict[str, bool] = Field(default_factory=dict)


class StepLog(BaseModel):
    """One model interaction (including retries of the same step)."""

    step_index: int
    retry_index: int = 0
    observation: str = Field(description="Exact model-facing input for this step.")
    model_output: str = ""
    parsed_action: dict[str, Any] | None = None
    action_valid: bool = True
    verifier: VerifierResult | None = None
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class RunRecord(BaseModel):
    """One (condition, task, seed) attempt."""

    run_id: str
    experiment_id: str = ""
    condition_id: str
    task_id: str
    seed: int
    model_profile: str
    harness_profile: str
    success: bool = False
    error_type: ErrorType = ""
    escalated: bool = False
    final_answer: dict[str, Any] | None = None
    final_verifier: VerifierResult | None = None
    steps: list[StepLog] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    retries: int = 0
    invalid_actions: int = 0
    verifier_failures: int = 0
    started_at: str = ""
    finished_at: str = ""
    notes: str = ""


def append_jsonl(path: str | Path, record: RunRecord) -> None:
    """Append one record to a JSONL file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def load_jsonl(path: str | Path) -> list[RunRecord]:
    """Load all records from a JSONL file."""
    records: list[RunRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(RunRecord.model_validate(json.loads(line)))
    return records
