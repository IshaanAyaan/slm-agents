"""Deterministic file/code navigation benchmark generator.

Produces real navigation tasks over real Python repositories with **unambiguous,
automatically-derived ground truth** — no human labeling and no API calls.

Method: parse every ``.py`` file with the stdlib ``ast`` module and index symbol
definitions (top-level classes, functions, and UPPER_CASE module constants). Keep only
symbols whose name is defined in **exactly one file** across the repo, so "which file
defines `X`?" has a single correct answer that the deterministic scorer can check.

Each repo is tagged with a split. Train-split repos feed the distillation pipeline;
held-out test-split repos measure generalization to code the SLM never trained on.

Usage:
    python generate_tasks.py --repo <path> --repo-id requests --split train \
        --max-tasks 200 --out suite_train.json [--append]
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

UPPER_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
# Names too generic to make an unambiguous, interesting task.
STOPNAMES = {
    "main", "run", "setup", "test", "config", "Config", "Error", "Base", "Meta",
    "wrapper", "inner", "decorator", "handler", "callback", "execute", "process",
    "get", "set", "update", "create", "delete", "load", "save", "parse", "build",
}


@dataclass
class Definition:
    """One symbol definition site."""

    name: str
    relpath: str
    lineno: int
    kind: str  # "class" | "function" | "constant"


def _iter_py_files(root: Path) -> list[Path]:
    skip = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist",
            ".tox", ".mypy_cache", ".pytest_cache", "tests", "test"}
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def collect_definitions(root: Path) -> list[Definition]:
    """Index top-level class/function/constant definitions across a repo."""
    defs: list[Definition] = []
    for path in _iter_py_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:  # top-level only -> stable, findable by grep
            if isinstance(node, ast.ClassDef):
                defs.append(Definition(node.name, rel, node.lineno, "class"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(Definition(node.name, rel, node.lineno, "function"))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and UPPER_RE.match(target.id):
                        defs.append(Definition(target.id, rel, node.lineno, "constant"))
    return defs


def unique_definitions(defs: list[Definition]) -> list[Definition]:
    """Keep symbols defined in exactly one file (unambiguous ground truth)."""
    by_name: dict[str, set[str]] = defaultdict(set)
    keep_first: dict[str, Definition] = {}
    for d in defs:
        by_name[d.name].add(d.relpath)
        keep_first.setdefault(d.name, d)
    out = []
    for name, files in by_name.items():
        if len(files) != 1:
            continue
        d = keep_first[name]
        if name in STOPNAMES or name.startswith("_"):
            continue
        if d.kind == "function" and len(name) < 5:
            continue
        out.append(d)
    return out


_PROMPTS = {
    "class": "Find the file in this repository where `class {name}` is defined and report its path.",
    "function": "Locate the definition of the function `{name}` and report which file contains it.",
    "constant": "Which file declares the module-level constant `{name}`?",
}


def build_suite(
    root: Path,
    repo_id: str,
    split: str,
    max_tasks: int,
    seed: int,
    workspace_str: str | None = None,
) -> list[dict]:
    """Build a balanced, reproducible list of NavTask dicts for one repo.

    ``workspace_str`` overrides the stored workspace path (e.g. a path relative to
    the suite file, for portability across checkouts).
    """
    workspace = workspace_str if workspace_str is not None else str(root)
    uniq = unique_definitions(collect_definitions(root))
    by_kind: dict[str, list[Definition]] = defaultdict(list)
    for d in uniq:
        by_kind[d.kind].append(d)
    rng = random.Random(f"{repo_id}:{seed}")
    for lst in by_kind.values():
        lst.sort(key=lambda d: (d.relpath, d.name))
        rng.shuffle(lst)

    # Round-robin across kinds for balance.
    order = ["class", "function", "constant"]
    picked: list[Definition] = []
    idx = {k: 0 for k in order}
    while len(picked) < max_tasks and any(idx[k] < len(by_kind[k]) for k in order):
        for k in order:
            if idx[k] < len(by_kind[k]):
                picked.append(by_kind[k][idx[k]])
                idx[k] += 1
                if len(picked) >= max_tasks:
                    break

    tasks = []
    for i, d in enumerate(picked):
        tasks.append(
            {
                "task_id": f"{repo_id}-{d.kind[:3]}-{i:04d}",
                "workspace": workspace,
                "prompt": _PROMPTS[d.kind].format(name=d.name),
                "expected_files": [d.relpath],
                "verification": {"method": "expected_files", "require_all_files": True},
                "allowed_tools": ["read_file", "grep", "glob"],
                "split": split,
                "metadata": {
                    "repo_id": repo_id,
                    "symbol": d.name,
                    "kind": d.kind,
                    "def_line": d.lineno,
                },
            }
        )
    return tasks


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="path to a repo checkout")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite-id", default=None)
    parser.add_argument("--append", action="store_true", help="merge into existing --out suite")
    parser.add_argument(
        "--workspace-rel",
        action="store_true",
        help="store workspace as a path relative to the --out suite file (portable across checkouts)",
    )
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    out_path = Path(args.out)
    import os as _os

    workspace_str = None
    if args.workspace_rel:
        workspace_str = _os.path.relpath(root, out_path.resolve().parent)
    tasks = build_suite(
        root, args.repo_id, args.split, args.max_tasks, args.seed, workspace_str
    )

    if args.append and out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        existing_ids = {t["task_id"] for t in existing["tasks"]}
        existing["tasks"].extend(t for t in tasks if t["task_id"] not in existing_ids)
        suite = existing
    else:
        suite = {
            "suite_id": args.suite_id or f"realbench-{args.split}",
            "description": "Deterministic file/code navigation tasks over real repos "
            "(unique-symbol ground truth, no labeling, no API).",
            "tasks": tasks,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    n_repos = len({t["metadata"]["repo_id"] for t in suite["tasks"]})
    print(f"{out_path}: {len(suite['tasks'])} tasks across {n_repos} repo(s) "
          f"(+{len(tasks)} from {args.repo_id})")


if __name__ == "__main__":
    main()
