"""Make `openharness` importable when running from a repo checkout.

The repo installs `openharness` from ``src/`` via hatch; for research scripts and
tests run straight from a checkout we prepend ``<repo>/src`` to ``sys.path`` so no
installation step is required and the existing package is reused unmodified.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def ensure_openharness_importable() -> None:
    """Prepend the repo's src/ directory to sys.path if needed."""
    try:
        import openharness  # noqa: F401
    except ModuleNotFoundError:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))


ensure_openharness_importable()
