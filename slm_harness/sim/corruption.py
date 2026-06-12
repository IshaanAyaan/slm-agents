"""Reproducible corruptions whose inverse is known by construction.

Each corruptor takes a clean artifact plus a seeded RNG and returns the corrupted
form along with a tag naming the corruption kind. Because we start from the clean
artifact, the oracle label is exact and no model ever judges correctness.
"""

from __future__ import annotations

import json
import random
import re


# -- JSON corruption (json_repair) -------------------------------------------------

def _single_quotes(text: str, rng: random.Random) -> str:
    return text.replace('"', "'")

def _trailing_comma(text: str, rng: random.Random) -> str:
    return re.sub(r"\}$", ",}", re.sub(r"\]([,}\s]*)$", r",]\1", text, count=1), count=1)

def _unquoted_keys(text: str, rng: random.Random) -> str:
    return re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', r"\1:", text)

def _python_literals(text: str, rng: random.Random) -> str:
    out = text
    for a, b in (("true", "True"), ("false", "False"), ("null", "None")):
        out = out.replace(a, b)
    return out

def _markdown_fence(text: str, rng: random.Random) -> str:
    lang = rng.choice(["json", "", "javascript"])
    return f"```{lang}\n{text}\n```"

def _leading_prose(text: str, rng: random.Random) -> str:
    prose = rng.choice([
        "Sure! Here is the action:",
        "The next action should be:",
        "I'll search for that now.",
        "Here's my response in the requested format:",
    ])
    return f"{prose}\n{text}"

def _truncate_tail(text: str, rng: random.Random) -> str:
    # Only ever drop closing structure ("}", "]", quotes, whitespace) so the
    # original object stays exactly recoverable from what remains.
    n = 0
    while n < len(text) and text[-(n + 1)] in '}]" \n' and n < 4:
        n += 1
    if n == 0:
        return text
    return text[:-rng.randint(1, n)]

def _missing_comma(text: str, rng: random.Random) -> str:
    positions = [m.start() for m in re.finditer(r',\s*"', text)]
    if not positions:
        return text
    pos = rng.choice(positions)
    return text[:pos] + text[pos + 1:]

def _extra_field_quote(text: str, rng: random.Random) -> str:
    # Drop the closing quote of one string value.
    matches = [m for m in re.finditer(r': "([^"]+)"', text)]
    if not matches:
        return text
    m = rng.choice(matches)
    end = m.end() - 1
    return text[:end] + text[end + 1:]


JSON_CORRUPTORS: dict[str, callable] = {
    "single_quotes": _single_quotes,
    "trailing_comma": _trailing_comma,
    "unquoted_keys": _unquoted_keys,
    "python_literals": _python_literals,
    "markdown_fence": _markdown_fence,
    "leading_prose": _leading_prose,
    "truncate_tail": _truncate_tail,
    "missing_comma": _missing_comma,
    "unterminated_string": _extra_field_quote,
}


def corrupt_json(obj: dict, rng: random.Random, n_layers: int = 1) -> tuple[str, list[str]]:
    """Corrupt a JSON object with 1..n random corruption layers."""
    text = json.dumps(obj, separators=(", ", ": "))
    kinds = rng.sample(sorted(JSON_CORRUPTORS), k=min(n_layers, len(JSON_CORRUPTORS)))
    # Apply structural corruptions before wrappers so fences stay on the outside.
    kinds.sort(key=lambda k: k in ("markdown_fence", "leading_prose"))
    for kind in kinds:
        text = JSON_CORRUPTORS[kind](text, rng)
    return text, kinds


# -- path corruption (path_normalizer) ----------------------------------------------

def _backslashes(path: str, rng: random.Random) -> str:
    return path.replace("/", "\\")

def _dotted_module(path: str, rng: random.Random) -> str:
    out = path[:-3] if path.endswith(".py") else path
    return out.replace("/", ".")

def _drop_extension(path: str, rng: random.Random) -> str:
    return path[:-3] if path.endswith(".py") else path

def _basename_only(path: str, rng: random.Random) -> str:
    return path.rsplit("/", 1)[-1]

def _partial_suffix(path: str, rng: random.Random) -> str:
    parts = path.split("/")
    if len(parts) <= 2:
        return path
    keep = rng.randint(2, len(parts) - 1)
    return "/".join(parts[-keep:])

def _leading_dotslash(path: str, rng: random.Random) -> str:
    return "./" + path

def _abs_prefix(path: str, rng: random.Random) -> str:
    prefix = rng.choice(["/home/user/project/", "/workspace/", "C:\\repo\\", "~/code/repo/"])
    sep = "\\" if "\\" in prefix else "/"
    return prefix + path.replace("/", sep)

def _typo(path: str, rng: random.Random) -> str:
    stem = path.rsplit("/", 1)[-1]
    if len(stem) < 6:
        return path
    i = rng.randint(len(path) - len(stem), len(path) - 2)
    ch = path[i]
    if not ch.isalpha():
        return path
    swap = "abcdefghijklmnopqrstuvwxyz".replace(ch.lower(), "")
    return path[:i] + rng.choice(swap) + path[i + 1:]

def _case_mangle(path: str, rng: random.Random) -> str:
    stem, _, ext = path.rpartition(".")
    return (stem.title().replace("_", "") + "." + ext) if stem else path


PATH_CORRUPTORS: dict[str, callable] = {
    "backslashes": _backslashes,
    "dotted_module": _dotted_module,
    "drop_extension": _drop_extension,
    "basename_only": _basename_only,
    "partial_suffix": _partial_suffix,
    "leading_dotslash": _leading_dotslash,
    "abs_prefix": _abs_prefix,
    "typo": _typo,
    "case_mangle": _case_mangle,
}


def corrupt_path(path: str, rng: random.Random) -> tuple[str, str]:
    """Apply one random path corruption; returns (noisy_mention, kind)."""
    kind = rng.choice(sorted(PATH_CORRUPTORS))
    return PATH_CORRUPTORS[kind](path, rng), kind


# -- identifier mangling (search_query_gen) ------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def identifier_words(name: str) -> list[str]:
    """Split snake_case / CamelCase identifiers into lowercase words."""
    parts: list[str] = []
    for chunk in name.split("_"):
        if not chunk:
            continue
        parts.extend(p for p in _CAMEL_RE.split(chunk) if p)
    return [p.lower() for p in parts]
