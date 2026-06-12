"""search_hit_ranker: pick the definition site out of real grep hits.

Hits are produced by actually scanning the repo for the symbol, so distractors are
genuine imports/usages/docstring mentions. Oracle = the AST-indexed definition site.
"""

from __future__ import annotations

import json
import random
import re

from slm_harness.sim.repo_pool import RepoIndex
from slm_harness.subroutines.base import Subroutine, extract_json_object
from slm_harness.subroutines.search_query_gen import _RepoText

MAX_HITS = 10

_SYSTEM = (
    "You rank code-search results. The user gives a target symbol and a numbered "
    "list of grep hits (path, line number, line text). Choose the hit showing the "
    "symbol's DEFINITION (its `def`/`class` statement or constant assignment), not "
    "an import, call, or mention.\n"
    'Reply with ONLY strict JSON: {"choice": <hit index>}'
)


def _grep_symbol(idx: RepoIndex, name: str) -> list[dict]:
    """All lines in the repo containing the symbol name (word-bounded)."""
    rx = re.compile(rf"\b{re.escape(name)}\b")
    hits: list[dict] = []
    for rel, text in _RepoText.files(str(idx.root)).items():
        for ln, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append({"path": rel, "line": ln, "text": line.strip()[:160]})
                if len(hits) > 400:
                    return hits
    return hits


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """One example per (symbol, shuffle): the def hit hidden among real usages."""
    rng = random.Random(f"search_hit_ranker:{split}:{seed}")
    out: list[dict] = []
    pool = [(idx, sym) for idx in indexes for sym in idx.symbols]
    rng.shuffle(pool)
    i = 0
    while len(out) < n and pool:
        idx, sym = pool[i % len(pool)]
        i += 1
        if i > 4 * len(pool):  # exhausted: not enough grep-able symbols
            break
        hits = _grep_symbol(idx, sym.name)
        def_hits = [h for h in hits if h["path"] == sym.relpath
                    and h["line"] == sym.lineno]
        others = [h for h in hits if not (h["path"] == sym.relpath
                                          and h["line"] == sym.lineno)]
        if not def_hits or len(others) < 3:
            continue
        rng.shuffle(others)
        chosen = others[: MAX_HITS - 1] + def_hits[:1]
        rng.shuffle(chosen)
        answer = chosen.index(def_hits[0])
        out.append({
            "id": f"search_hit_ranker-{split}-{len(out):05d}",
            "subroutine": "search_hit_ranker",
            "split": split,
            "input": {
                "query": sym.name,
                "hits": [{"i": j, **h} for j, h in enumerate(chosen)],
            },
            "target": {"choice": answer},
            "meta": {"repo": idx.repo_id, "symbol": sym.name, "kind": sym.kind},
        })
    return out


class SearchHitRanker(Subroutine):
    name = "search_hit_ranker"
    description = "Pick the definition site among real grep hits for a symbol."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        q = example["input"]["query"]
        lines = [f'[{h["i"]}] {h["path"]}:{h["line"]}: {h["text"]}'
                 for h in example["input"]["hits"]]
        return f"Symbol: {q}\nHits:\n" + "\n".join(lines)

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        choice = payload.get("choice")
        if not isinstance(choice, int) or isinstance(choice, bool):
            return None, "'choice' must be an integer hit index"
        return {"choice": choice}, ""

    def verify(self, example: dict, output: dict) -> bool:
        return output["choice"] == example["target"]["choice"]

    def rules_baseline(self, example: dict) -> dict | None:
        """Prefer def/class/assignment-shaped lines; penalize tests and imports."""
        q = re.escape(example["input"]["query"])
        best, best_score = None, -1e9
        for h in example["input"]["hits"]:
            text, score = h["text"], 0.0
            if re.match(rf"(async\s+)?def {q}\b|class {q}\b|{q}\s*(:.*)?=", text):
                score += 10
            if re.match(r"from\s|import\s", text):
                score -= 5
            if "test" in h["path"].lower():
                score -= 3
            score -= h["line"] * 1e-6
            if score > best_score:
                best, best_score = h["i"], score
        return {"choice": best} if best is not None else None
