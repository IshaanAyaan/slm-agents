"""Coarse MCP tools backed by the subroutine harness.

Deployment has no oracle, so success is enforced with *guards*: invariants that
are checkable without ground truth (schema validity, candidate membership, span
bounds, regex compilability, index range). The pipeline per call:

    render -> specialist backend -> parse -> guard -> (one retry) -> rules fallback

Backends:
    rules  deterministic baselines only (no model; default, runs anywhere)
    stub   echo backend for protocol tests
    hf     a local fine-tuned checkpoint per subroutine (models_root env)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from slm_harness.subroutines.registry import REGISTRY

# -- deployment-time guards (no oracle needed) --------------------------------------

def _guard_json_repair(inp: dict, out: dict) -> str:
    return ""  # schema validation in parse_output is the guard


def _guard_action_router(inp: dict, out: dict) -> str:
    return ""  # closed label set enforced by parse_output


def _guard_path_normalizer(inp: dict, out: dict) -> str:
    if out["path"] is not None and out["path"] not in inp["candidates"]:
        return "path is not one of the offered candidates"
    return ""


def _guard_search_query_gen(inp: dict, out: dict) -> str:
    try:
        re.compile(out["pattern"])
    except re.error as exc:
        return f"pattern does not compile: {exc}"
    if len(out["pattern"]) < 3:
        return "pattern too short to be selective"
    return ""


def _guard_search_hit_ranker(inp: dict, out: dict) -> str:
    if not 0 <= out["choice"] < len(inp["hits"]):
        return f"choice {out['choice']} outside hit range"
    return ""


def _guard_read_span_selector(inp: dict, out: dict) -> str:
    if not (inp["window_start"] <= out["start"] <= out["end"] <= inp["window_end"]):
        return "span outside the shown window"
    return ""


def _guard_evidence_judge(inp: dict, out: dict) -> str:
    if out["decision"] == "answer":
        shown = {s["path"] for s in inp["evidence"]}
        if out["path"] not in shown:
            return "answer path was never shown in evidence"
    return ""


def _guard_trace_localizer(inp: dict, out: dict) -> str:
    if out["path"] not in inp["traceback"]:
        base = out["path"].rsplit("/", 1)[-1]
        if base not in inp["traceback"]:
            return "culprit path does not appear in the traceback"
    return ""


GUARDS: dict[str, Callable[[dict, dict], str]] = {
    "json_repair": _guard_json_repair,
    "action_router": _guard_action_router,
    "path_normalizer": _guard_path_normalizer,
    "search_query_gen": _guard_search_query_gen,
    "search_hit_ranker": _guard_search_hit_ranker,
    "read_span_selector": _guard_read_span_selector,
    "evidence_judge": _guard_evidence_judge,
    "trace_localizer": _guard_trace_localizer,
}


# -- specialist backends -------------------------------------------------------------

class RulesBackend:
    """Deterministic baselines as the zero-dependency reference backend."""

    name = "rules"

    def complete(self, subroutine: str, example: dict) -> str:
        out = REGISTRY[subroutine].rules_baseline(example)
        if out is None:
            return ""
        return json.dumps(out, separators=(",", ":"))


class StubBackend:
    """Returns a fixed reply; for protocol tests."""

    name = "stub"

    def __init__(self, reply: str = "{}"):
        self.reply = reply

    def complete(self, subroutine: str, example: dict) -> str:
        return self.reply


class HFBackend:
    """Local fine-tuned specialist checkpoints (one per subroutine)."""

    name = "hf"

    def __init__(self, models_root: str, size: str = "qwen2.5-0.5b"):
        self.models_root = models_root
        self.size = size
        self._runners: dict[str, Any] = {}

    def complete(self, subroutine: str, example: dict) -> str:
        from slm_harness.evals.subroutine_eval import HFRunner

        if subroutine not in self._runners:
            self._runners[subroutine] = HFRunner(
                os.path.join(self.models_root, subroutine, self.size),
                batch_size=1)
        sub = REGISTRY[subroutine]
        chat = [{"role": "system", "content": sub.system_prompt()},
                {"role": "user", "content": sub.render_user(example)}]
        texts, _ = self._runners[subroutine].generate([chat], max_new_tokens=220)
        return texts[0]


def make_backend() -> Any:
    kind = os.environ.get("SLM_BACKEND", "rules")
    if kind == "rules":
        return RulesBackend()
    if kind == "stub":
        return StubBackend()
    if kind == "hf":
        return HFBackend(os.environ["SLM_MODELS_ROOT"],
                         os.environ.get("SLM_SIZE", "qwen2.5-0.5b"))
    raise ValueError(f"unknown SLM_BACKEND: {kind}")


# -- the single coarse entry point ---------------------------------------------------

def run_specialist_task(subroutine: str, payload: dict,
                        backend: Any | None = None) -> dict:
    """Validate -> specialist -> parse -> guard -> retry once -> rules fallback."""
    if subroutine not in REGISTRY:
        return {"ok": False, "error": f"unknown subroutine '{subroutine}'",
                "known": sorted(REGISTRY)}
    sub = REGISTRY[subroutine]
    backend = backend or make_backend()
    example = {"id": "mcp-call", "subroutine": subroutine, "split": "deploy",
               "input": payload, "target": {}, "meta": {}}

    attempts = []
    for attempt in (1, 2):
        try:
            raw = backend.complete(subroutine, example)
        except Exception as exc:  # backend failure must not crash the server
            attempts.append({"attempt": attempt, "error": f"backend: {exc}"})
            break
        out, err = sub.parse_output(raw)
        if out is None:
            attempts.append({"attempt": attempt, "error": f"schema: {err}"})
            continue
        guard_err = GUARDS[subroutine](payload, out)
        if guard_err:
            attempts.append({"attempt": attempt, "error": f"guard: {guard_err}"})
            continue
        return {"ok": True, "subroutine": subroutine, "output": out,
                "backend": backend.name, "attempts": attempt,
                "verified": "schema+guard"}

    # Fallback: deterministic rules, still guard-checked.
    rules_out = sub.rules_baseline(example)
    if rules_out is not None and not GUARDS[subroutine](payload, rules_out):
        return {"ok": True, "subroutine": subroutine, "output": rules_out,
                "backend": "rules-fallback", "attempts": attempts,
                "verified": "schema+guard"}
    return {"ok": False, "subroutine": subroutine, "attempts": attempts,
            "error": "specialist and rules fallback both failed"}


# -- MCP tool surface (few coarse tools, many subroutines behind them) ----------------

TOOL_DEFS: list[dict] = [
    {
        "name": "slm_run_specialist_task",
        "description": ("Run one narrow developer subroutine on a small verified "
                        "specialist. Subroutines: " + ", ".join(sorted(REGISTRY))),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subroutine": {"type": "string", "enum": sorted(REGISTRY)},
                "payload": {"type": "object"},
            },
            "required": ["subroutine", "payload"],
        },
    },
    {
        "name": "slm_repair_action",
        "description": "Repair a malformed JSON tool-action string into strict JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {"raw": {"type": "string"}},
            "required": ["raw"],
        },
    },
    {
        "name": "slm_normalize_path",
        "description": "Resolve a noisy file-path mention against candidate paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mention": {"type": "string"},
                "candidates": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["mention", "candidates"],
        },
    },
    {
        "name": "slm_localize_failure",
        "description": "Find the culprit project frame and category in a traceback.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "traceback": {"type": "string"},
                "project_prefix": {"type": "string"},
            },
            "required": ["traceback", "project_prefix"],
        },
    },
]


def call_tool(name: str, arguments: dict, backend: Any | None = None) -> dict:
    """Dispatch one MCP tools/call to the harness."""
    if name == "slm_run_specialist_task":
        return run_specialist_task(arguments["subroutine"],
                                   arguments["payload"], backend)
    alias = {
        "slm_repair_action": "json_repair",
        "slm_normalize_path": "path_normalizer",
        "slm_localize_failure": "trace_localizer",
    }
    if name in alias:
        return run_specialist_task(alias[name], arguments, backend)
    return {"ok": False, "error": f"unknown tool '{name}'"}
