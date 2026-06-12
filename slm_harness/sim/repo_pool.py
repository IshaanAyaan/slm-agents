"""Shallow-clone pool of real Python repos + AST symbol indexes.

Train/val data comes from TRAIN_REPOS; the test split comes only from TEST_REPOS,
so every reported number measures generalization to repos the model never saw.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

from slm_harness.tasks.realbench.generate_tasks import (
    Definition,
    collect_definitions,
    unique_definitions,
)

TRAIN_REPOS: dict[str, str] = {
    "flask": "https://github.com/pallets/flask",
    "click": "https://github.com/pallets/click",
    "rich": "https://github.com/Textualize/rich",
}
TEST_REPOS: dict[str, str] = {
    "httpx": "https://github.com/encode/httpx",
    "jinja2": "https://github.com/pallets/jinja",
}
ALL_REPOS = {**TRAIN_REPOS, **TEST_REPOS}

# Pinned for reproducibility (release tags current as of 2026-06).
REPO_REFS: dict[str, str] = {
    "flask": "3.1.0",
    "click": "8.1.8",
    "rich": "v13.9.4",
    "httpx": "0.28.1",
    "jinja2": "3.1.5",
}


@dataclass
class RepoIndex:
    """One repo checkout plus its oracle symbol index."""

    repo_id: str
    root: Path
    files: list[str]                 # repo-relative .py paths
    symbols: list[Definition]        # unique-file symbol definitions
    split: str                       # "train" | "test"


def ensure_repo(repo_id: str, cache_dir: Path) -> Path:
    """Clone (depth-1 at the pinned ref) or reuse a cached checkout."""
    if repo_id not in ALL_REPOS:
        raise KeyError(f"unknown repo: {repo_id}")
    dest = cache_dir / repo_id
    if (dest / ".git").is_dir():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--quiet", "--depth", "1",
            "--branch", REPO_REFS[repo_id], ALL_REPOS[repo_id], str(dest),
        ],
        check=True,
    )
    return dest


def _py_files(root: Path) -> list[str]:
    skip = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist",
            ".tox", ".mypy_cache", ".pytest_cache"}
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part in skip for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return out


def load_index(repo_id: str, cache_dir: Path) -> RepoIndex:
    """Build the oracle index for one repo."""
    root = ensure_repo(repo_id, cache_dir)
    split = "train" if repo_id in TRAIN_REPOS else "test"
    symbols = sorted(
        unique_definitions(collect_definitions(root)),
        key=lambda d: (d.relpath, d.lineno, d.name),
    )
    return RepoIndex(repo_id=repo_id, root=root, files=_py_files(root),
                     symbols=symbols, split=split)


def definition_span(root: Path, definition: Definition) -> tuple[int, int] | None:
    """Oracle (start, end) 1-based line span of a definition, decorators included."""
    path = root / definition.relpath
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != definition.name or node.lineno != definition.lineno:
                continue
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return start, int(node.end_lineno or node.lineno)
        if isinstance(node, ast.Assign) and node.lineno == definition.lineno:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == definition.name:
                    return node.lineno, int(node.end_lineno or node.lineno)
    return None
