"""evidence_judge: decide whether gathered evidence already answers the question.

The judge sees a navigation question plus snippets the agent has read so far.
Positive cases include the real definition snippet; negative cases contain only
genuine look-alikes (imports, usages, re-exports of the same symbol). Oracle =
whether the AST definition span is among the snippets.
"""

from __future__ import annotations

import json
import random
import re

from slm_harness.sim.repo_pool import RepoIndex, definition_span
from slm_harness.subroutines.base import Subroutine, extract_json_object
from slm_harness.subroutines.search_query_gen import _RepoText

_SYSTEM = (
    "You are the answer-readiness check of a code-navigation agent. The user shows "
    "the question and the evidence snippets read so far. If the evidence already "
    "contains the symbol's actual definition, answer with the defining file. "
    "Otherwise tell the agent to continue.\n"
    'Reply with ONLY strict JSON: {"decision": "answer", "path": "<file>"} or '
    '{"decision": "continue"}'
)

_QUESTIONS = {
    "class": "Which file defines the class `{name}`?",
    "function": "Which file contains the definition of the function `{name}`?",
    "constant": "Which file declares the constant `{name}`?",
}


def _snippet(idx: RepoIndex, path: str, center: int, rng: random.Random,
             width: int = 8) -> dict | None:
    text = _RepoText.files(str(idx.root)).get(path)
    if text is None:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    lo = max(1, center - rng.randint(1, 3))
    hi = min(len(lines), lo + width)
    body = "\n".join(lines[i - 1] for i in range(lo, hi + 1))
    return {"path": path, "line_start": lo, "line_end": hi, "text": body[:800]}


def _usage_sites(idx: RepoIndex, name: str, def_path: str) -> list[tuple[str, int]]:
    """Lines mentioning the symbol outside its definition file."""
    rx = re.compile(rf"\b{re.escape(name)}\b")
    sites = []
    for rel, text in _RepoText.files(str(idx.root)).items():
        if rel == def_path:
            continue
        for ln, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                sites.append((rel, ln))
    return sites


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Half answerable (definition snippet present), half decoys only."""
    rng = random.Random(f"evidence_judge:{split}:{seed}")
    out: list[dict] = []
    pool = [(idx, sym) for idx in indexes for sym in idx.symbols]
    rng.shuffle(pool)
    i = 0
    while len(out) < n and pool:
        idx, sym = pool[i % len(pool)]
        i += 1
        if i > 4 * len(pool):
            break
        usages = _usage_sites(idx, sym.name, sym.relpath)
        if len(usages) < 2:
            continue
        answerable = rng.random() < 0.5
        snippets: list[dict] = []
        rng.shuffle(usages)
        for rel, ln in usages[: rng.randint(2, 4)]:
            s = _snippet(idx, rel, ln, rng)
            if s:
                snippets.append(s)
        if answerable:
            span = definition_span(idx.root, sym)
            if span is None:
                continue
            s = _snippet(idx, sym.relpath, sym.lineno, rng,
                         width=min(12, span[1] - span[0] + 4))
            if s is None:
                continue
            snippets.append(s)
        if len(snippets) < 2:
            continue
        rng.shuffle(snippets)
        out.append({
            "id": f"evidence_judge-{split}-{len(out):05d}",
            "subroutine": "evidence_judge",
            "split": split,
            "input": {
                "question": _QUESTIONS[sym.kind].format(name=sym.name),
                "evidence": snippets,
            },
            "target": ({"decision": "answer", "path": sym.relpath}
                       if answerable else {"decision": "continue"}),
            "meta": {"repo": idx.repo_id, "symbol": sym.name, "kind": sym.kind},
        })
    return out


class EvidenceJudge(Subroutine):
    name = "evidence_judge"
    description = "Decide answer-vs-continue from gathered evidence snippets."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        parts = [f"Question: {example['input']['question']}", "Evidence:"]
        for s in example["input"]["evidence"]:
            parts.append(f"--- {s['path']} lines {s['line_start']}-{s['line_end']} ---")
            parts.append(s["text"])
        return "\n".join(parts)

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        decision = payload.get("decision")
        if decision == "continue":
            return {"decision": "continue"}, ""
        if decision == "answer":
            path = payload.get("path")
            if not isinstance(path, str) or not path:
                return None, "'answer' decision requires a 'path' string"
            return {"decision": "answer", "path": path}, ""
        return None, "'decision' must be 'answer' or 'continue'"

    def verify(self, example: dict, output: dict) -> bool:
        return output == example["target"]

    def rules_baseline(self, example: dict) -> dict | None:
        """def/class/assignment regex over snippets (fooled by look-alikes)."""
        q = example["input"]["question"]
        m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", q)
        if not m:
            return {"decision": "continue"}
        name = re.escape(m.group(1))
        rx = re.compile(rf"^\s*(async\s+)?(def|class)\s+{name}\b|^{name}\s*(:.*)?=",
                        re.MULTILINE)
        for s in example["input"]["evidence"]:
            if rx.search(s["text"]):
                return {"decision": "answer", "path": s["path"]}
        return {"decision": "continue"}
