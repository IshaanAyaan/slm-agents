"""SLM-subagent research scaffold.

Tests the claim: a fine-tuned small language model with a co-designed, task-specific
harness, used as a subagent under a fixed frontier-model orchestrator, beats a cheap
general-purpose subagent on cost-per-successful-task while preserving success rate.

See README.md and REPO_MAP.md in this directory.
"""

from slm_harness import _bootstrap  # noqa: F401  (side effect: sys.path)

__version__ = "0.1.0"
