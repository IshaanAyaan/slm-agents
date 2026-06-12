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


# -- Phase-2: synthetic mini-repo with known symbols (offline, no cloning) ----------

_MINI_FILES = {
    "src/app/config_loader.py": '''\
"""Configuration loading."""
import os

DEFAULT_TIMEOUT_SECONDS = 30


def load_config_file(path):
    """Read a config file.

    Multi-line docstring with
dedented content to fool indentation rules.
    """
    data = open(path).read()
    return parse_config_text(data)


def parse_config_text(text):
    return {line.split("=")[0]: line.split("=")[1]
            for line in text.splitlines() if "=" in line}
''',
    "src/app/retry_budget.py": '''\
from src.app.config_loader import DEFAULT_TIMEOUT_SECONDS, load_config_file


class RetryBudgetTracker:
    """Tracks remaining retries."""

    def __init__(self, limit):
        self.limit = limit

    def consume_retry_token(self):
        self.limit -= 1
        return self.limit >= 0
''',
    "src/app/span_utils.py": '''\
import re

from src.app.retry_budget import RetryBudgetTracker


def normalize_span_bounds(start, end):
    sample = """
not actually code
    """
    return (min(start, end), max(start, end))


def annotated_helper_function(x):
    return x * 2
''',
    "src/app/render_engine.py": '''\
from src.app.span_utils import normalize_span_bounds


class TemplateRenderEngine:
    def render(self, template):
        bounds = normalize_span_bounds(0, len(template))
        return template[bounds[0]:bounds[1]]
''',
    "src/app/json_codec.py": '''\
import json

from src.app.config_loader import parse_config_text

MAX_PAYLOAD_BYTES = 65536


def encode_payload_safely(obj):
    return json.dumps(obj)
''',
    "src/cli/main_entry.py": '''\
from src.app.render_engine import TemplateRenderEngine
from src.app.json_codec import encode_payload_safely


def dispatch_command_line(argv):
    engine = TemplateRenderEngine()
    return engine.render(str(argv))
''',
    "src/cli/output_writer.py": '''\
from src.app.json_codec import MAX_PAYLOAD_BYTES


def write_formatted_output(text):
    assert len(text) < MAX_PAYLOAD_BYTES
    return text
''',
    "src/util/path_helpers.py": '''\
def split_repo_relative_path(path):
    return path.split("/")
''',
    "src/util/cache_store.py": '''\
class PersistentCacheStore:
    def lookup_cached_entry(self, key):
        return None
''',
    "src/util/hash_tools.py": '''\
import hashlib

from src.util.cache_store import PersistentCacheStore


def stable_content_digest(blob):
    store = PersistentCacheStore()
    assert store.lookup_cached_entry(blob) is None
    return hashlib.sha256(blob).hexdigest()
''',
    "src/net/socket_pool.py": '''\
from src.app.retry_budget import RetryBudgetTracker


class PooledSocketManager:
    def acquire_pooled_socket(self):
        tracker = RetryBudgetTracker(3)
        while tracker.consume_retry_token():
            pass
        return None
''',
    "src/net/url_parsing.py": '''\
from src.util.path_helpers import split_repo_relative_path


def canonicalize_request_url(url):
    parts = split_repo_relative_path(url)
    return "/".join(p for p in parts if p)
''',
    "tests/test_config_loader.py": '''\
from src.app.config_loader import load_config_file, parse_config_text


def test_parse_config_text():
    assert parse_config_text("a=1") == {"a": "1"}
''',
    "tests/test_render_engine.py": '''\
from src.app.render_engine import TemplateRenderEngine


def test_render():
    assert TemplateRenderEngine().render("ab") == "ab"
''',
}


@pytest.fixture(scope="session")
def mini_repo(tmp_path_factory) -> Path:
    """Synthetic repo with cross-file imports/usages and known unique symbols."""
    root = tmp_path_factory.mktemp("minirepo")
    for rel, text in _MINI_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def mini_index(mini_repo: Path):
    """RepoIndex over the mini repo, built without git."""
    from slm_harness.sim.repo_pool import RepoIndex, _py_files
    from slm_harness.tasks.realbench.generate_tasks import (
        collect_definitions,
        unique_definitions,
    )

    symbols = sorted(
        unique_definitions(collect_definitions(mini_repo)),
        key=lambda d: (d.relpath, d.lineno, d.name),
    )
    return RepoIndex(repo_id="minirepo", root=mini_repo,
                     files=_py_files(mini_repo), symbols=symbols, split="train")
