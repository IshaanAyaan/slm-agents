"""Externalized state for the custom search/navigation harness.

Requirement: the SLM carries almost no free-form context. All durable state —
goal, search history, discovered files, read spans, failed attempts, verifier
feedback, budgets — lives here, outside the model. Each step the model sees only
a minimal JSON observation derived from this state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from research.slm_harness.schemas.runlog import StepLog


@dataclass
class SearchRecord:
    """One executed search."""

    pattern: str
    file_glob: str
    n_hits: int


@dataclass
class Hit:
    """One search hit (workspace-relative path)."""

    path: str
    line: int
    text: str


@dataclass
class ReadRecord:
    """One executed read."""

    path: str
    offset: int
    limit: int
    n_lines: int


@dataclass
class CustomHarnessState:
    """All task state, externalized from the model."""

    goal: str
    max_steps: int
    step: int = 0
    searches: list[SearchRecord] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    reads: list[ReadRecord] = field(default_factory=list)
    known_files: list[str] = field(default_factory=list)  # ordered, deduped
    failed_actions: list[str] = field(default_factory=list)
    feedback: str = ""  # verifier feedback for the *current* (retried) step only
    last_result: dict[str, Any] = field(default_factory=dict)

    MAX_HITS_SHOWN: int = 8
    MAX_FILES_SHOWN: int = 10

    # ----- state transitions -------------------------------------------------

    def record_search(self, pattern: str, file_glob: str, hits: list[Hit]) -> None:
        """Fold a completed search into state."""
        self.searches.append(SearchRecord(pattern=pattern, file_glob=file_glob, n_hits=len(hits)))
        for hit in hits:
            self.hits.append(hit)
            if hit.path not in self.known_files:
                self.known_files.append(hit.path)
        self.last_result = {
            "type": "search",
            "pattern": pattern,
            "n_hits": len(hits),
            "hits": [
                {"f": h.path, "l": h.line, "t": h.text[:120]} for h in hits[: self.MAX_HITS_SHOWN]
            ],
        }

    def record_read(self, path: str, offset: int, limit: int, content: str, n_lines: int) -> None:
        """Fold a completed read into state."""
        self.reads.append(ReadRecord(path=path, offset=offset, limit=limit, n_lines=n_lines))
        if path not in self.known_files:
            self.known_files.append(path)
        self.last_result = {
            "type": "read",
            "f": path,
            "o": offset,
            "n": n_lines,
            "content": content[:1200],
        }

    def record_failure(self, description: str) -> None:
        """Record a failed/invalid attempt and surface it as feedback."""
        self.failed_actions.append(description[:200])
        self.feedback = description[:300]

    def clear_feedback(self) -> None:
        """Clear per-step feedback after a successful step."""
        self.feedback = ""

    # ----- action-space constraint (requirement 1) ---------------------------

    def valid_actions(self) -> list[str]:
        """Only actions valid for the current step are exposed to the model."""
        actions = ["SEARCH"]
        if self.known_files:
            actions.append("READ")
        if self.reads:  # answering requires having read evidence
            actions.append("ANSWER")
        actions.append("ESCALATE")
        return actions

    # ----- minimal structured observation (requirement 2) --------------------

    def build_observation(self, char_budget: int = 2000) -> str:
        """Render the minimal JSON observation for the model."""
        obs: dict[str, Any] = {
            "goal": self.goal,
            "step": self.step,
            "steps_left": max(0, self.max_steps - self.step),
            "valid_actions": self.valid_actions(),
            "searches": [
                {"pattern": s.pattern, "glob": s.file_glob, "hits": s.n_hits}
                for s in self.searches[-4:]
            ],
            "files": self.known_files[: self.MAX_FILES_SHOWN],
            "reads": [{"f": r.path, "o": r.offset, "n": r.n_lines} for r in self.reads[-4:]],
            "last_result": self.last_result,
        }
        if self.feedback:
            obs["feedback"] = self.feedback
        text = json.dumps(obs, separators=(",", ":"), ensure_ascii=False)
        if len(text) > char_budget:
            # Degrade gracefully: shrink the largest field (last_result content).
            lr = dict(obs.get("last_result", {}))
            if "content" in lr:
                overshoot = len(text) - char_budget
                lr["content"] = lr["content"][: max(100, len(lr["content"]) - overshoot)]
                obs["last_result"] = lr
                text = json.dumps(obs, separators=(",", ":"), ensure_ascii=False)
        return text[: char_budget + 200]


@dataclass
class HarnessOutcome:
    """Normalized result of one harness run on one task."""

    success: bool
    escalated: bool = False
    error_type: str = ""
    final_answer: dict[str, Any] | None = None
    final_verifier: dict[str, Any] | None = None
    steps: list[StepLog] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    retries: int = 0
    invalid_actions: int = 0
    verifier_failures: int = 0
    notes: str = ""
