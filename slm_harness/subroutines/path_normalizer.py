"""path_normalizer: resolve a noisy path mention to a canonical repo file (or none).

Mentions come from corrupting real repo paths (dotted modules, backslashes, typos,
absolute prefixes, bare basenames). The harness supplies a candidate shortlist; the
specialist picks the exact canonical path or abstains. Oracle = the pre-corruption
path; "none" cases corrupt a path that is then withheld from the candidates.
"""

from __future__ import annotations

import difflib
import json
import random

from slm_harness.sim.corruption import corrupt_path
from slm_harness.sim.repo_pool import RepoIndex
from slm_harness.subroutines.base import Subroutine, extract_json_object

_SYSTEM = (
    "You normalize noisy file-path mentions. The user gives a mention and a list of "
    "real repository file paths. Pick the candidate the mention refers to, or null "
    "if none of them match.\n"
    'Reply with ONLY strict JSON: {"path": "<candidate exactly as listed>"} or '
    '{"path": null}'
)


def _candidates(true_path: str, files: list[str], rng: random.Random,
                include_true: bool, k: int = 10) -> list[str]:
    """The true path plus confusable distractors, shuffled."""
    pool = [f for f in files if f != true_path]
    base = true_path.rsplit("/", 1)[-1]
    same_base = [f for f in pool if f.rsplit("/", 1)[-1] == base]
    close = difflib.get_close_matches(true_path, pool, n=k, cutoff=0.3)
    distractors: list[str] = []
    for f in same_base + close + rng.sample(pool, min(len(pool), k)):
        if f not in distractors:
            distractors.append(f)
    picked = distractors[: k - 1 if include_true else k]
    if include_true:
        picked.append(true_path)
    rng.shuffle(picked)
    return picked


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Corrupt real paths; ~15% of examples have no correct candidate."""
    rng = random.Random(f"path_normalizer:{split}:{seed}")
    out: list[dict] = []
    while len(out) < n:
        idx = rng.choice(indexes)
        if len(idx.files) < 12:
            continue
        true_path = rng.choice(idx.files)
        mention, kind = corrupt_path(true_path, rng)
        if mention == true_path and rng.random() < 0.7:
            continue  # keep a few identity mentions, drop most
        include_true = rng.random() >= 0.15
        cands = _candidates(true_path, idx.files, rng, include_true)
        out.append({
            "id": f"path_normalizer-{split}-{len(out):05d}",
            "subroutine": "path_normalizer",
            "split": split,
            "input": {"mention": mention, "candidates": cands},
            "target": {"path": true_path if include_true else None},
            "meta": {"repo": idx.repo_id, "corruption": kind,
                     "true_path": true_path},
        })
    return out


def _normalize(mention: str) -> str:
    """Canonicalize separators/prefixes so suffix comparison is meaningful."""
    m = mention.strip().replace("\\", "/")
    for prefix in ("./", "~/", "/"):
        while m.startswith(prefix):
            m = m[len(prefix):]
    if "/" not in m and "." in m and not m.endswith(".py"):
        m = m.replace(".", "/")  # dotted module form
    if not m.endswith(".py"):
        m += ".py"
    return m.lower()


class PathNormalizer(Subroutine):
    name = "path_normalizer"
    description = "Resolve a noisy path mention to a canonical repo path or abstain."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        lines = "\n".join(f"- {c}" for c in example["input"]["candidates"])
        return f"Mention: {example['input']['mention']}\nCandidates:\n{lines}"

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        if "path" not in payload:
            return None, "missing required field 'path'"
        path = payload["path"]
        if path is not None and not isinstance(path, str):
            return None, "'path' must be a string or null"
        return {"path": path}, ""

    def verify(self, example: dict, output: dict) -> bool:
        return output["path"] == example["target"]["path"]

    def rules_baseline(self, example: dict) -> dict | None:
        """Suffix-overlap scoring with a difflib tie-break; abstain when weak."""
        mention = _normalize(example["input"]["mention"])
        best, best_score = None, 0.0
        for cand in example["input"]["candidates"]:
            c = cand.lower()
            score = 0.0
            if c.endswith(mention) or mention.endswith(c):
                score = 1.0 + len(mention) / max(len(c), 1)
            else:
                score = difflib.SequenceMatcher(None, mention, c).ratio()
            if score > best_score:
                best, best_score = cand, score
        if best_score < 0.6:
            return {"path": None}
        return {"path": best}
