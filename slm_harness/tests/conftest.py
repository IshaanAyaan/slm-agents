"""Test bootstrap: make `research` and `openharness` importable from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURE_REPO = REPO_ROOT / "slm_harness/tasks/smoke/fixture_repo"
SMOKE_TASKS = REPO_ROOT / "slm_harness/tasks/smoke/smoke_tasks.json"


@pytest.fixture()
def fixture_repo() -> Path:
    """Path to the synthetic navigation workspace."""
    assert FIXTURE_REPO.is_dir()
    return FIXTURE_REPO


@pytest.fixture()
def smoke_tasks_path() -> Path:
    """Path to the smoke task suite JSON."""
    assert SMOKE_TASKS.is_file()
    return SMOKE_TASKS
