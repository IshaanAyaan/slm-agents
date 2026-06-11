"""Make the repo's packages importable when running from a checkout.

The repo ships two top-level packages. `slm_harness` is the research code and
`openharness` is a small vendored subset of the OpenHarness tool layer (see
NOTICE.md). Running scripts straight from a checkout works without an install
step because we prepend the repo root to sys.path here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_openharness_importable() -> None:
    """Prepend the repo root to sys.path if the packages are not installed."""
    try:
        import openharness  # noqa: F401
    except ModuleNotFoundError:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))


ensure_openharness_importable()
