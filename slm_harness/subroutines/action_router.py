"""action_router: map a messy natural-language progress report to the next action.

The harness keeps the real state machine; in deployment a master agent (or log
renderer) describes progress in free text. The specialist must recover the next
action under the harness policy. Labels come from the deterministic policy applied
to the underlying state features, never from the rendered text.
"""

from __future__ import annotations

import json
import random

from slm_harness.sim.repo_pool import RepoIndex
from slm_harness.subroutines.base import Subroutine, extract_json_object

ACTIONS = ["SEARCH", "READ", "ANSWER", "ESCALATE"]

_SYSTEM = (
    "You are the routing policy of a code-navigation subagent. Given a progress "
    "report, decide the next action.\n"
    "Policy: SEARCH if no search has been run yet, or the last search found nothing "
    "and at least 2 steps remain. READ if there are unread search hits, or hits were "
    "read but the definition has not been found and at least 2 steps remain. ANSWER "
    "only once the definition itself has been read. ESCALATE when steps are exhausted "
    "(fewer than 2 remain) without finding the definition.\n"
    'Reply with ONLY strict JSON: {"next_action":"SEARCH"|"READ"|"ANSWER"|"ESCALATE"}'
)


def policy(state: dict) -> str:
    """Deterministic harness routing policy (the oracle labeler)."""
    if not state["searched"]:
        return "SEARCH"
    if state["found_definition"]:
        return "ANSWER"
    if state["steps_left"] < 2:
        return "ESCALATE"
    if state["n_hits"] == 0:
        return "SEARCH"
    return "READ"


_OPENERS = [
    "Status update on the navigation task for `{sym}`:",
    "Progress so far while hunting for {sym}:",
    "Here is where the subagent stands on locating `{sym}`.",
    "Report ({repo} workspace):",
]
_NO_SEARCH = [
    "No searches have been run yet.",
    "The agent just started and has not queried the workspace at all.",
    "Nothing has been executed so far; the tool log is empty.",
]
_SEARCH_HITS = [
    "The last search returned {n} matching locations.",
    "Grep produced {n} hits across the repo.",
    "{n} candidate lines came back from the most recent search.",
]
_SEARCH_EMPTY = [
    "The last search came back empty.",
    "Zero hits were returned for the most recent pattern.",
    "The previous query matched nothing in the workspace.",
]
_READ_MISS = [
    "It read {path} but the definition was not in that span.",
    "A read of {path} showed only usages, not the definition.",
    "{path} was inspected; no `def`/`class` for the target there.",
]
_READ_HIT = [
    "It read {path} and the span clearly contains the definition of {sym}.",
    "The definition of {sym} was found while reading {path}.",
    "{path} lines around {line} show `class {sym}` / `def {sym}` itself.",
]
_BUDGET = [
    "{k} steps remain in the budget.",
    "The remaining step budget is {k}.",
    "Budget: {k} more actions allowed.",
]
_DISTRACTORS = [
    "The user seems to be in a hurry.",
    "Workspace indexing finished 2 minutes ago.",
    "Token usage is well within limits.",
    "The repository was cloned at depth 1.",
    "Earlier, an unrelated lint warning was printed.",
]

# Held-out phrasings used ONLY for the test split: the rules baseline (and any
# template-memorizing model) was never shown these, so the test measures
# generalization to unseen report wordings, not template recall.
_TEST_TEMPLATES = {
    "openers": [
        "Quick recap of the hunt for {sym} inside {repo}:",
        "Current situation regarding `{sym}`:",
        "Where things stand with the `{sym}` lookup --",
    ],
    "no_search": [
        "So far the workspace has not been touched; no queries went out.",
        "Still at square one: not a single search executed.",
    ],
    "search_hits": [
        "Most recently, scanning surfaced {n} places that mention the target.",
        "A pattern scan turned up {n} candidate locations.",
    ],
    "search_empty": [
        "The latest scan turned up nothing at all.",
        "No locations were surfaced by the last pattern.",
    ],
    "read_miss": [
        "{path} got opened, yet the body shown had calls only -- the target is declared elsewhere.",
        "After opening {path}, the agent saw references but never the declaration itself.",
    ],
    "read_hit": [
        "Opening {path} revealed the actual declaration of {sym} right there in the span.",
        "{path} turned out to hold the real declaration -- `{sym}` is created at line {line}.",
    ],
    "budget": [
        "Allowance left before forced stop: {k}.",
        "The agent can take {k} further turns.",
    ],
}


def _render(state: dict, idx: RepoIndex, rng: random.Random,
            heldout: bool = False) -> str:
    sym, repo = state["symbol"], idx.repo_id
    path = state["read_path"]
    t = _TEST_TEMPLATES
    openers = t["openers"] if heldout else _OPENERS
    no_search = t["no_search"] if heldout else _NO_SEARCH
    search_hits = t["search_hits"] if heldout else _SEARCH_HITS
    search_empty = t["search_empty"] if heldout else _SEARCH_EMPTY
    read_miss = t["read_miss"] if heldout else _READ_MISS
    read_hit = t["read_hit"] if heldout else _READ_HIT
    budget = t["budget"] if heldout else _BUDGET
    parts = [rng.choice(openers).format(sym=sym, repo=repo)]
    if not state["searched"]:
        parts.append(rng.choice(no_search))
    elif state["n_hits"] == 0:
        parts.append(rng.choice(search_empty))
    else:
        parts.append(rng.choice(search_hits).format(n=state["n_hits"]))
        if state["read_any"]:
            tmpl = read_hit if state["found_definition"] else read_miss
            parts.append(rng.choice(tmpl).format(path=path, sym=sym,
                                                 line=state["def_line"]))
    parts.append(rng.choice(budget).format(k=state["steps_left"]))
    if rng.random() < 0.5:
        parts.insert(rng.randint(1, len(parts) - 1), rng.choice(_DISTRACTORS))
    return " ".join(parts)


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Sample states covering every policy branch, render them naturalistically."""
    rng = random.Random(f"action_router:{split}:{seed}")
    out: list[dict] = []
    while len(out) < n:
        idx = rng.choice(indexes)
        sym = rng.choice(idx.symbols)
        searched = rng.random() < 0.8
        n_hits = 0 if (searched and rng.random() < 0.3) else rng.randint(1, 40)
        read_any = searched and n_hits > 0 and rng.random() < 0.7
        found = read_any and rng.random() < 0.5
        state = {
            "symbol": sym.name,
            "searched": searched,
            "n_hits": n_hits if searched else 0,
            "read_any": read_any,
            "found_definition": found,
            "read_path": sym.relpath if found else rng.choice(idx.files),
            "def_line": sym.lineno,
            "steps_left": rng.randint(0, 8),
        }
        label = policy(state)
        out.append({
            "id": f"action_router-{split}-{len(out):05d}",
            "subroutine": "action_router",
            "split": split,
            "input": {"state_text": _render(state, idx, rng,
                                            heldout=split == "test")},
            "target": {"next_action": label},
            "meta": {"repo": idx.repo_id, "state": state},
        })
    return out


class ActionRouter(Subroutine):
    name = "action_router"
    description = "Choose the next harness action from a noisy progress report."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        return example["input"]["state_text"]

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        choice = payload.get("next_action")
        if choice not in ACTIONS:
            return None, f"next_action must be one of {ACTIONS}"
        return {"next_action": choice}, ""

    def verify(self, example: dict, output: dict) -> bool:
        return output["next_action"] == example["target"]["next_action"]

    def rules_baseline(self, example: dict) -> dict | None:
        """Regex feature extraction from the report text, then the policy."""
        import re

        text = example["input"]["state_text"].lower()
        searched = not any(p in text for p in
                           ["no searches have been run", "has not queried",
                            "tool log is empty", "nothing has been executed"])
        empty = any(p in text for p in ["came back empty", "zero hits",
                                        "matched nothing"])
        found = any(p in text for p in ["contains the definition",
                                        "definition of", "show `class"])
        miss = any(p in text for p in ["was not in that span", "only usages",
                                       "no `def`/`class`"])
        m = re.search(r"(\d+) (?:steps remain|more actions)", text)
        m2 = re.search(r"budget is (\d+)|budget: (\d+)", text)
        steps = int(m.group(1)) if m else int(next(g for g in m2.groups() if g)) if m2 else 4
        state = {"searched": searched, "n_hits": 0 if empty else 5,
                 "found_definition": found and not miss, "steps_left": steps}
        return {"next_action": policy(state)}
