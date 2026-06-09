"""Strict JSON action schema for the custom search/navigation harness.

The custom harness exposes exactly four actions. The model must reply with a single
JSON object; anything else is an invalid action (retryable, with feedback).
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class SearchAction(BaseModel):
    """SEARCH(pattern, file_glob, root): regex content search over the workspace."""

    action: Literal["SEARCH"] = "SEARCH"
    pattern: str = Field(min_length=1, max_length=400)
    file_glob: str = Field(default="**/*", max_length=200)
    root: str = Field(default=".", max_length=400)


class ReadAction(BaseModel):
    """READ(path, offset, limit): read a line range of one file."""

    action: Literal["READ"] = "READ"
    path: str = Field(min_length=1, max_length=500)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=80, ge=1, le=400)


class EvidenceSpan(BaseModel):
    """A cited span supporting the answer."""

    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


class AnswerAction(BaseModel):
    """ANSWER(files, evidence, confidence): final answer returned to the orchestrator."""

    action: Literal["ANSWER"] = "ANSWER"
    files: list[str] = Field(min_length=1, max_length=20)
    evidence: list[EvidenceSpan] = Field(default_factory=list, max_length=20)
    answer_text: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EscalateAction(BaseModel):
    """ESCALATE(reason): give up and hand back to the orchestrator."""

    action: Literal["ESCALATE"] = "ESCALATE"
    reason: str = Field(min_length=1, max_length=1000)


SubagentAction = Annotated[
    Union[SearchAction, ReadAction, AnswerAction, EscalateAction],
    Field(discriminator="action"),
]

_ACTION_ADAPTER: TypeAdapter[SubagentAction] = TypeAdapter(SubagentAction)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(raw: str) -> tuple[SubagentAction | None, str]:
    """Parse a model reply into an action.

    Returns (action, "") on success or (None, error_message) on failure.
    Tolerates surrounding prose/code fences but requires exactly one JSON object.
    """
    if not raw or not raw.strip():
        return None, "empty model output; reply with one JSON action object"
    text = raw.strip()
    # Strip a markdown code fence if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None, "no JSON object found in output"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at pos {exc.pos}"
    if not isinstance(payload, dict):
        return None, "JSON payload must be an object"
    if "action" not in payload:
        return None, "missing required field 'action'"
    try:
        return _ACTION_ADAPTER.validate_python(payload), ""
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", ()))
        return None, f"schema error at '{loc}': {first.get('msg', 'invalid')}"
