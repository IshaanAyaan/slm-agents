"""search_query_gen: turn a natural-language ask into an executable search pattern.

The request paraphrases an identifier as plain words ("the retry budget checker");
the specialist must emit a regex that, when actually executed over the repo, surfaces
the file defining that identifier. Success is execution-verified: the oracle file
must rank in the top-K matched files. The teacher never judges anything.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from slm_harness.sim.corruption import identifier_words
from slm_harness.sim.repo_pool import RepoIndex
from slm_harness.subroutines.base import Subroutine, extract_json_object

TOP_K = 5
_MAX_FILE_MATCHES = 200

_SYSTEM = (
    "You write code-search regexes. The user describes, in plain words, a Python "
    "symbol they need to find in a repository. Produce one concise regex (Python "
    "re syntax, case-insensitive search) that will match lines near that symbol's "
    "definition. Prefer reconstructing the likely identifier from the words.\n"
    'Reply with ONLY strict JSON: {"pattern": "<regex>"}'
)

_TEMPLATES = {
    "function": [
        "Find the function that handles {words} in this repo.",
        "Where is the {words} routine defined?",
        "Locate the helper responsible for {words}.",
        "I need the file with the function for {words}.",
    ],
    "class": [
        "Find the class that represents {words}.",
        "Where is the {words} class implemented?",
        "Locate the type used for {words}.",
        "Which file holds the class handling {words}?",
    ],
    "constant": [
        "Find the module-level constant for {words}.",
        "Where is the {words} constant declared?",
        "Locate the global setting for {words}.",
    ],
}

# Held-out phrasings used ONLY for the test split (generalization, not recall).
_TEST_ONLY_TEMPLATES = {
    "function": [
        "Somewhere in here a function takes care of {words}; track it down.",
        "Hunt down whichever helper does {words}.",
        "Point me at the code implementing {words}, please.",
    ],
    "class": [
        "There's a class modelling {words} somewhere; find it.",
        "Track down the object type behind {words}.",
        "Show me where the {words} abstraction lives.",
    ],
    "constant": [
        "A top-level value configures {words}; hunt it down.",
        "Track down the hardcoded default for {words}.",
    ],
}


class _RepoText:
    """In-memory file contents for execution verification (built lazily once)."""

    _cache: dict[str, dict[str, str]] = {}

    @classmethod
    def files(cls, root: str) -> dict[str, str]:
        if root not in cls._cache:
            contents: dict[str, str] = {}
            rootp = Path(root)
            for p in rootp.rglob("*.py"):
                rel = p.relative_to(rootp).as_posix()
                if any(part in {".git", "__pycache__", ".venv", "venv"}
                       for part in rel.split("/")):
                    continue
                try:
                    contents[rel] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
            cls._cache[root] = contents
        return cls._cache[root]


def execute_pattern(pattern: str, repo_root: str) -> list[str] | None:
    """Run the regex over the repo; files ranked by match count (None = bad regex)."""
    if not pattern or len(pattern) < 3 or len(pattern) > 200:
        return None
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None
    counts: list[tuple[int, str]] = []
    for rel, text in _RepoText.files(repo_root).items():
        n = 0
        for _ in rx.finditer(text):
            n += 1
            if n >= _MAX_FILE_MATCHES:
                break
        if n:
            counts.append((-n, rel))
    counts.sort()
    return [rel for _, rel in counts]


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Paraphrase unique symbols as word-level asks; gold pattern = the identifier."""
    rng = random.Random(f"search_query_gen:{split}:{seed}")
    out: list[dict] = []
    pool: list[tuple[RepoIndex, object]] = []
    for idx in indexes:
        for sym in idx.symbols:
            if len(identifier_words(sym.name)) >= 2:  # one-word names are trivial
                pool.append((idx, sym))
    rng.shuffle(pool)
    i = 0
    while len(out) < n and pool:
        idx, sym = pool[i % len(pool)]
        i += 1
        words = " ".join(identifier_words(sym.name))
        bank = _TEST_ONLY_TEMPLATES if split == "test" else _TEMPLATES
        tmpl = rng.choice(bank[sym.kind])
        gold_pattern = re.escape(sym.name)
        ranked = execute_pattern(gold_pattern, str(idx.root))
        if ranked is None or sym.relpath not in ranked[:TOP_K]:
            continue  # gold must itself pass execution verification
        out.append({
            "id": f"search_query_gen-{split}-{len(out):05d}",
            "subroutine": "search_query_gen",
            "split": split,
            "input": {"request": tmpl.format(words=words)},
            "target": {"pattern": gold_pattern},
            "meta": {"repo": idx.repo_id, "symbol": sym.name, "kind": sym.kind,
                     "def_path": sym.relpath, "repo_root": str(idx.root)},
        })
    return out


class SearchQueryGen(Subroutine):
    name = "search_query_gen"
    description = "Write an executable search regex from a plain-words request."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        return example["input"]["request"]

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        pattern = payload.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return None, "'pattern' must be a non-empty string"
        return {"pattern": pattern.strip()}, ""

    def verify(self, example: dict, output: dict) -> bool:
        """Execution check: oracle def file must be in the top-K matched files."""
        ranked = execute_pattern(output["pattern"], example["meta"]["repo_root"])
        if ranked is None:
            return False
        return example["meta"]["def_path"] in ranked[:TOP_K]

    def rules_baseline(self, example: dict) -> dict | None:
        """Join the request's content words with underscores (snake_case guess)."""
        text = example["input"]["request"].lower().rstrip("?.!")
        # Generic English/meta stopwords, deliberately NOT tuned to any template
        # bank: function words plus search-y meta verbs only.
        stop = {"find", "the", "that", "this", "in", "is", "a", "an", "of", "for",
                "with", "where", "which", "file", "files", "repo", "repository",
                "function", "class", "constant", "module", "level", "defined",
                "declared", "implemented", "located", "locate", "i", "need",
                "holds", "used", "to", "handles", "handling", "responsible",
                "routine", "helper", "type", "global", "setting", "represents",
                "somewhere", "here", "there", "it", "me", "please", "show",
                "point", "at", "track", "hunt", "down", "whichever", "takes",
                "care", "does", "lives", "behind", "modelling", "modeling"}
        words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in stop]
        if not words:
            return None
        return {"pattern": "_".join(words)}
