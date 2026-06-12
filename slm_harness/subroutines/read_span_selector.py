"""read_span_selector: report the exact line span of a definition inside a window.

The harness shows a numbered window of real file content around (but not aligned
with) the definition. The specialist returns the absolute 1-based [start, end] span
covering the whole definition, decorators included. Oracle = AST lineno/end_lineno,
which naive indentation rules get wrong on multiline strings and decorators.
"""

from __future__ import annotations

import json
import random
import re

from slm_harness.sim.repo_pool import RepoIndex, definition_span
from slm_harness.subroutines.base import Subroutine, extract_json_object
from slm_harness.subroutines.search_query_gen import _RepoText

WINDOW = 110          # lines shown to the model
START_SLACK = 1       # allowed |pred_start - gold_start|
END_SLACK = 2         # allowed |pred_end - gold_end|

_SYSTEM = (
    "You locate definition spans. The user gives a numbered excerpt of a Python "
    "file and a symbol defined inside it. Reply with the absolute line span of the "
    "complete definition: from its first decorator (or the def/class/assignment "
    "line if undecorated) through the last line of its body.\n"
    'Reply with ONLY strict JSON: {"start": <int>, "end": <int>}'
)


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Windows jittered around oracle spans from real files."""
    rng = random.Random(f"read_span_selector:{split}:{seed}")
    out: list[dict] = []
    pool = [(idx, sym) for idx in indexes for sym in idx.symbols
            if sym.kind in ("class", "function")]
    rng.shuffle(pool)
    i = 0
    while len(out) < n and pool:
        idx, sym = pool[i % len(pool)]
        i += 1
        if i > 4 * len(pool):
            break
        span = definition_span(idx.root, sym)
        if span is None:
            continue
        start, end = span
        if end - start + 1 > WINDOW - 10:
            continue  # definition must fit in the window with context
        text = _RepoText.files(str(idx.root)).get(sym.relpath)
        if text is None:
            continue
        lines = text.splitlines()
        lead = rng.randint(3, max(4, WINDOW - (end - start + 1) - 3))
        w_start = max(1, start - lead)
        w_end = min(len(lines), w_start + WINDOW - 1)
        if end > w_end:
            continue
        numbered = "\n".join(
            f"{ln:5d}| {lines[ln - 1]}" for ln in range(w_start, w_end + 1)
        )
        out.append({
            "id": f"read_span_selector-{split}-{len(out):05d}",
            "subroutine": "read_span_selector",
            "split": split,
            "input": {"path": sym.relpath, "symbol": sym.name, "kind": sym.kind,
                      "window_start": w_start, "window_end": w_end,
                      "window": numbered},
            "target": {"start": start, "end": end},
            "meta": {"repo": idx.repo_id, "def_line": sym.lineno},
        })
    return out


class ReadSpanSelector(Subroutine):
    name = "read_span_selector"
    description = "Report the exact line span of a symbol's definition in a window."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        inp = example["input"]
        return (f"File: {inp['path']}\nSymbol: `{inp['symbol']}` ({inp['kind']})\n"
                f"Excerpt (lines {inp['window_start']}-{inp['window_end']}):\n"
                f"{inp['window']}")

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        start, end = payload.get("start"), payload.get("end")
        for v in (start, end):
            if not isinstance(v, int) or isinstance(v, bool):
                return None, "'start' and 'end' must be integers"
        if start < 1 or end < start:
            return None, "need 1 <= start <= end"
        return {"start": start, "end": end}, ""

    def verify(self, example: dict, output: dict) -> bool:
        gold_s, gold_e = example["target"]["start"], example["target"]["end"]
        return (abs(output["start"] - gold_s) <= START_SLACK
                and abs(output["end"] - gold_e) <= END_SLACK)

    def rules_baseline(self, example: dict) -> dict | None:
        """Indentation scan: correct until multiline strings/comments fool it."""
        inp = example["input"]
        lines = [l.split("| ", 1)[1] if "| " in l else ""
                 for l in inp["window"].splitlines()]
        w_start = inp["window_start"]
        name = re.escape(inp["symbol"])
        def_rx = re.compile(rf"^(\s*)(async\s+)?(def|class)\s+{name}\b")
        def_i = next((i for i, l in enumerate(lines) if def_rx.match(l)), None)
        if def_i is None:
            return None
        indent = len(def_rx.match(lines[def_i]).group(1))
        start = def_i
        while start > 0 and lines[start - 1].lstrip().startswith("@"):
            start -= 1
        end = def_i
        for j in range(def_i + 1, len(lines)):
            stripped = lines[j].strip()
            if not stripped:
                continue
            if len(lines[j]) - len(lines[j].lstrip()) <= indent:
                break
            end = j
        return {"start": w_start + start, "end": w_start + end}
