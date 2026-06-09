"""Task/eval schema for file/code navigation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

VerificationMethod = Literal["expected_files", "answer_regex", "both"]


class VerificationSpec(BaseModel):
    """How a task's final answer is scored (deterministically)."""

    method: VerificationMethod = "expected_files"
    require_all_files: bool = Field(
        default=True,
        description="If True, every expected file must be cited; else any one suffices.",
    )
    forbid_extra_files: bool = Field(
        default=False, description="If True, citing files outside expected_files fails the task."
    )
    answer_regex: str | None = Field(
        default=None, description="Regex the free-text answer must match (method answer_regex/both)."
    )


class NavTask(BaseModel):
    """One file/code navigation task."""

    task_id: str
    workspace: str = Field(
        description="Path to the repo fixture / workspace root, relative to the task file."
    )
    prompt: str = Field(description="The orchestrator's query to the subagent.")
    expected_files: list[str] = Field(
        default_factory=list,
        description="Target files, as workspace-relative POSIX paths.",
    )
    verification: VerificationSpec = Field(default_factory=VerificationSpec)
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["read_file", "grep", "glob"],
        description="Generic-harness tool allowlist for this task.",
    )
    split: Literal["train", "val", "test"] = "test"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TaskSuite(BaseModel):
    """A set of tasks plus suite-level metadata."""

    suite_id: str
    tasks: list[NavTask]
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "TaskSuite":
        """Load a suite from JSON and resolve workspace paths relative to the file."""
        p = Path(path).resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        suite = cls.model_validate(data)
        for task in suite.tasks:
            ws = Path(task.workspace)
            if not ws.is_absolute():
                task.workspace = str((p.parent / ws).resolve())
        return suite

    def by_split(self, split: str) -> list[NavTask]:
        """Return tasks in a given split."""
        return [t for t in self.tasks if t.split == split]
