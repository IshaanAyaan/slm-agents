"""Common interface for Phase-2 subroutines.

A subroutine bundles everything one narrow task needs: prompt rendering, output
parsing, a deterministic verifier, a rules-only baseline, and a data generator.
The same example dict flows through generation -> SFT export -> eval -> verification,
so the verifier is the single source of truth for "success" in every condition.

Example dict contract (per subroutine, stored as JSONL):
    {
      "id": str,                  # unique within the dataset
      "subroutine": str,          # registry key
      "split": "train"|"val"|"test",
      "input": {...},             # schema-bound input payload
      "target": {...},            # oracle output payload (never teacher-judged)
      "meta": {...},              # provenance (repo, symbol, corruption kind, ...)
    }
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str) -> tuple[dict | None, str]:
    """Pull the first JSON object out of a model reply (tolerates fences/prose)."""
    if not raw or not raw.strip():
        return None, "empty output"
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None, "no JSON object found"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at pos {exc.pos}"
    if not isinstance(payload, dict):
        return None, "JSON payload must be an object"
    return payload, ""


class Subroutine(ABC):
    """One narrow, verifiable developer-agent task."""

    #: registry key, e.g. "json_repair"
    name: str = ""
    #: one-line task description for the paper/registry table
    description: str = ""

    @abstractmethod
    def system_prompt(self) -> str:
        """Fixed system prompt used for SFT and eval."""

    @abstractmethod
    def render_user(self, example: dict) -> str:
        """Render example["input"] into the user turn."""

    @abstractmethod
    def render_target(self, example: dict) -> str:
        """Render example["target"] into the gold assistant turn (compact JSON)."""

    @abstractmethod
    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        """Parse + schema-validate a model reply. Returns (payload, error)."""

    @abstractmethod
    def verify(self, example: dict, output: dict) -> bool:
        """Deterministic success check of a parsed output against the oracle."""

    @abstractmethod
    def rules_baseline(self, example: dict) -> dict | None:
        """Best-effort deterministic (no-model) solver; None when it abstains."""

    # -- shared helpers -------------------------------------------------------

    def to_chat(self, example: dict) -> dict:
        """Render one example as a 3-turn chat row for SFT."""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": self.render_user(example)},
                {"role": "assistant", "content": self.render_target(example)},
            ]
        }

    def score(self, example: dict, raw: str) -> dict:
        """Parse + verify one raw model reply; returns the per-item record."""
        payload, err = self.parse_output(raw)
        valid = payload is not None
        ok = bool(valid and self.verify(example, payload))
        return {
            "id": example["id"],
            "schema_valid": valid,
            "success": ok,
            "error": err,
        }
