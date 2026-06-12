"""json_repair: recover a schema-valid harness action from a mangled model reply.

Input is a corrupted serialization of a valid SubagentAction (single quotes, prose,
fences, truncated braces, Python literals, ...). The specialist must emit the exact
original action as strict JSON. Oracle = the pre-corruption object; the verifier is
field-exact equality after schema validation.
"""

from __future__ import annotations

import json
import random
from typing import Any

from slm_harness.schemas.actions import parse_action
from slm_harness.sim.corruption import corrupt_json
from slm_harness.sim.repo_pool import RepoIndex
from slm_harness.subroutines.base import Subroutine, extract_json_object

_SYSTEM = (
    "You repair malformed JSON tool actions. The user gives you a broken reply that "
    "was meant to be one JSON action object with schema:\n"
    'SEARCH {"action","pattern","file_glob","root"} | '
    'READ {"action","path","offset","limit"} | '
    'ANSWER {"action","files","evidence","answer_text","confidence"} | '
    'ESCALATE {"action","reason"}\n'
    "Reply with ONLY the repaired strict-JSON object. Preserve every recoverable "
    "field value exactly. No prose, no code fences."
)


def _sample_action(idx: RepoIndex, rng: random.Random) -> dict[str, Any]:
    """A realistic, fully-explicit action dict using real symbols/paths."""
    sym = rng.choice(idx.symbols)
    kind = rng.choice(["SEARCH", "SEARCH", "READ", "READ", "ANSWER", "ESCALATE"])
    if kind == "SEARCH":
        pattern = rng.choice([sym.name, f"def {sym.name}", f"class {sym.name}",
                              f"{sym.name}\\("])
        return {"action": "SEARCH", "pattern": pattern,
                "file_glob": rng.choice(["**/*.py", "**/*", "src/**/*.py"]),
                "root": "."}
    if kind == "READ":
        offset = max(0, sym.lineno - rng.randint(1, 30))
        return {"action": "READ", "path": sym.relpath, "offset": offset,
                "limit": rng.choice([40, 80, 120])}
    if kind == "ANSWER":
        return {
            "action": "ANSWER",
            "files": [sym.relpath],
            "evidence": [{"path": sym.relpath, "line_start": sym.lineno,
                          "line_end": sym.lineno + rng.randint(0, 12)}],
            "answer_text": f"`{sym.name}` is defined in {sym.relpath}.",
            "confidence": rng.choice([0.6, 0.7, 0.8, 0.9, 0.95]),
        }
    return {"action": "ESCALATE",
            "reason": rng.choice([
                f"No definition of {sym.name} found after exhausting the step budget.",
                "Search returned no hits and the retry budget is spent.",
                f"The workspace does not appear to contain {sym.relpath}.",
            ])}


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Build n oracle-labeled repair examples from real-repo-flavored actions."""
    rng = random.Random(f"json_repair:{split}:{seed}")
    out: list[dict] = []
    while len(out) < n:
        idx = rng.choice(indexes)
        action = _sample_action(idx, rng)
        layers = rng.choice([1, 1, 1, 2, 2, 3])
        raw, kinds = corrupt_json(action, rng, n_layers=layers)
        if raw.strip() == json.dumps(action, separators=(", ", ": ")):
            continue  # corruption was a no-op; skip uninformative example
        out.append({
            "id": f"json_repair-{split}-{len(out):05d}",
            "subroutine": "json_repair",
            "split": split,
            "input": {"raw": raw},
            "target": action,
            "meta": {"repo": idx.repo_id, "corruptions": kinds},
        })
    return out


class JsonRepair(Subroutine):
    name = "json_repair"
    description = "Repair a mangled JSON tool action into the exact original object."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        return "Broken reply:\n" + example["input"]["raw"]

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        action, err = parse_action(raw)
        if action is None:
            return None, err
        return action.model_dump(), ""

    def verify(self, example: dict, output: dict) -> bool:
        gold, err = parse_action(json.dumps(example["target"]))
        if gold is None:  # defensive; targets are valid by construction
            raise ValueError(f"invalid gold action: {err}")
        return output == gold.model_dump()

    def rules_baseline(self, example: dict) -> dict | None:
        """The harness's own tolerant parser (fences/prose) — no model involved."""
        action, _ = parse_action(example["input"]["raw"])
        return action.model_dump() if action is not None else None
