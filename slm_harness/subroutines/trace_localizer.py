"""trace_localizer: name the culprit project frame and category from a traceback.

Tracebacks are synthesized from real repo frames (real paths, real source lines)
interleaved with stdlib/site-packages frames, sometimes as chained exceptions.
Oracle = the injected culprit frame (deepest project frame of the final chain) and
the category implied by the injected exception type.
"""

from __future__ import annotations

import json
import random
import re

from slm_harness.sim.repo_pool import RepoIndex
from slm_harness.subroutines.base import Subroutine, extract_json_object
from slm_harness.subroutines.search_query_gen import _RepoText

CATEGORIES = {
    "AttributeError": "null_attribute",
    "KeyError": "missing_key",
    "TypeError": "type_mismatch",
    "IndexError": "bad_index",
    "ValueError": "bad_value",
    "ImportError": "import_error",
    "ZeroDivisionError": "arithmetic",
}

_SYSTEM = (
    "You triage Python tracebacks for a repair agent. Identify the culprit frame "
    "(the deepest frame inside the PROJECT code, not stdlib or site-packages) and "
    "classify the failure.\n"
    "Categories: null_attribute, missing_key, type_mismatch, bad_index, bad_value, "
    "import_error, arithmetic.\n"
    'Reply with ONLY strict JSON: '
    '{"path": "<project-relative file>", "line": <int>, "category": "<category>"}'
)

_STDLIB_FRAMES = [
    ("/usr/lib/python3.11/json/decoder.py", 355, "obj, end = self.scan_once(s, idx)"),
    ("/usr/lib/python3.11/os.py", 225, "head, tail = path.split(name)"),
    ("/usr/lib/python3.11/re/__init__.py", 171, "return _compile(pattern, flags).search(string)"),
    ("/usr/lib/python3.11/pathlib.py", 542, "return cls._from_parts(args)"),
    ("/usr/lib/python3.11/typing.py", 379, "return func(*args, **kwds)"),
]
_SITE_FRAMES = [
    ("/usr/lib/python3/dist-packages/werkzeug/serving.py", 333, "execute(self.server.app)"),
    ("/usr/lib/python3/dist-packages/urllib3/connectionpool.py", 716, "httplib_response = self._make_request("),
    ("/usr/lib/python3/dist-packages/pluggy/_callers.py", 103, "res = hook_impl.function(*args)"),
]

_MESSAGES = {
    "AttributeError": ["'NoneType' object has no attribute '{attr}'",
                       "'dict' object has no attribute '{attr}'"],
    "KeyError": ["'{attr}'"],
    "TypeError": ["{attr}() missing 1 required positional argument: 'value'",
                  "unsupported operand type(s) for +: 'int' and 'str'",
                  "'NoneType' object is not callable"],
    "IndexError": ["list index out of range"],
    "ValueError": ["invalid literal for int() with base 10: '{attr}'",
                   "not enough values to unpack (expected 2, got 1)"],
    "ImportError": ["cannot import name '{attr}' from '{mod}'"],
    "ZeroDivisionError": ["division by zero"],
}


def _project_frames(idx: RepoIndex, rng: random.Random, k: int,
                    prefix: str) -> list[tuple[str, int, str, str]]:
    """k real (abs_path, line, code, rel_path) frames from the repo."""
    frames = []
    files = [f for f in idx.files if f.endswith(".py")]
    for _ in range(k * 4):
        rel = rng.choice(files)
        text = _RepoText.files(str(idx.root)).get(rel, "")
        lines = [(i, l) for i, l in enumerate(text.splitlines(), start=1)
                 if l.strip() and not l.strip().startswith("#")]
        if not lines:
            continue
        ln, code = rng.choice(lines)
        frames.append((prefix + rel, ln, code.strip()[:140], rel))
        if len(frames) == k:
            break
    return frames


def _render_block(frames: list[tuple[str, int, str]], exc: str, msg: str) -> str:
    out = ["Traceback (most recent call last):"]
    for path, ln, code in frames:
        out.append(f'  File "{path}", line {ln}, in <module>')
        out.append(f"    {code}")
    out.append(f"{exc}: {msg}")
    return "\n".join(out)


def generate(indexes: list[RepoIndex], split: str, n: int, seed: int) -> list[dict]:
    """Synthetic-but-real-framed tracebacks with known culprit + category."""
    rng = random.Random(f"trace_localizer:{split}:{seed}")
    out: list[dict] = []
    while len(out) < n:
        idx = rng.choice(indexes)
        prefix = rng.choice(["/home/user/project/", "/workspace/app/",
                             f"/srv/{idx.repo_id}/", "./"])
        proj = _project_frames(idx, rng, rng.randint(2, 4), prefix)
        if len(proj) < 2:
            continue
        exc = rng.choice(sorted(CATEGORIES))
        sym = rng.choice(idx.symbols)
        msg = rng.choice(_MESSAGES[exc]).format(attr=sym.name,
                                                mod=idx.repo_id)
        # Culprit = deepest project frame; sometimes stdlib/site frames sit below it.
        culprit_abs, culprit_line, culprit_code, culprit_rel = proj[-1]
        frames = [(p, l, c) for p, l, c, _ in proj]
        if rng.random() < 0.6:
            tail = rng.sample(_STDLIB_FRAMES + _SITE_FRAMES, rng.randint(1, 2))
            frames = frames + tail
        if rng.random() < 0.3:
            head = rng.sample(_SITE_FRAMES, 1)
            frames = head + frames
        block = _render_block(frames, exc, msg)
        if rng.random() < 0.25:
            pre_exc = rng.choice([e for e in sorted(CATEGORIES) if e != exc])
            pre_msg = rng.choice(_MESSAGES[pre_exc]).format(attr="config",
                                                            mod=idx.repo_id)
            decoy = _render_block(
                [(p, l, c) for p, l, c, _ in _project_frames(idx, rng, 2, prefix)],
                pre_exc, pre_msg)
            block = (decoy + "\n\nDuring handling of the above exception, "
                     "another exception occurred:\n\n" + block)
        out.append({
            "id": f"trace_localizer-{split}-{len(out):05d}",
            "subroutine": "trace_localizer",
            "split": split,
            "input": {"traceback": block, "project_prefix": prefix},
            "target": {"path": culprit_rel, "line": culprit_line,
                       "category": CATEGORIES[exc]},
            "meta": {"repo": idx.repo_id, "exception": exc},
        })
    return out


class TraceLocalizer(Subroutine):
    name = "trace_localizer"
    description = "Identify culprit project frame and failure category in a traceback."

    def system_prompt(self) -> str:
        return _SYSTEM

    def render_user(self, example: dict) -> str:
        return (f"Project root prefix: {example['input']['project_prefix']}\n\n"
                + example["input"]["traceback"])

    def render_target(self, example: dict) -> str:
        return json.dumps(example["target"], separators=(",", ":"))

    def parse_output(self, raw: str) -> tuple[dict | None, str]:
        payload, err = extract_json_object(raw)
        if payload is None:
            return None, err
        path, line, cat = payload.get("path"), payload.get("line"), payload.get("category")
        if not isinstance(path, str) or not path:
            return None, "'path' must be a non-empty string"
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            return None, "'line' must be a positive integer"
        if cat not in set(CATEGORIES.values()):
            return None, f"'category' must be one of {sorted(set(CATEGORIES.values()))}"
        return {"path": path, "line": line, "category": cat}, ""

    def verify(self, example: dict, output: dict) -> bool:
        gold = example["target"]
        # Accept the path with or without the project prefix.
        path = output["path"]
        prefix = example["input"]["project_prefix"]
        if path.startswith(prefix):
            path = path[len(prefix):]
        return (path == gold["path"] and output["line"] == gold["line"]
                and output["category"] == gold["category"])

    def rules_baseline(self, example: dict) -> dict | None:
        """Deepest frame under the project prefix in the last chain + exc mapping."""
        tb = example["input"]["traceback"]
        prefix = example["input"]["project_prefix"]
        last_chain = tb.split("During handling of the above exception")[-1]
        frames = re.findall(r'File "([^"]+)", line (\d+)', last_chain)
        culprit = None
        for path, ln in frames:
            if path.startswith(prefix):
                culprit = (path[len(prefix):], int(ln))
        m = re.search(r"^(\w+Error|ZeroDivisionError):", last_chain.strip().splitlines()[-1])
        if culprit is None or m is None:
            return None
        category = CATEGORIES.get(m.group(1))
        if category is None:
            return None
        return {"path": culprit[0], "line": culprit[1], "category": category}
