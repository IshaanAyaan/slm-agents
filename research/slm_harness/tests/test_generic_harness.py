"""Generic harness: tool loop, distractor tools, final-answer scoring."""

from __future__ import annotations

from pathlib import Path

from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock

from research.slm_harness.harness.generic import DISTRACTOR_TOOL_SCHEMAS, GenericHarness
from research.slm_harness.model_clients.stub import OracleStubClient, ScriptedClient
from research.slm_harness.schemas.config import HarnessProfile, ModelProfile, PricingConfig
from research.slm_harness.schemas.task import NavTask

PRICING = PricingConfig(usd_per_million_input=1.0, usd_per_million_output=2.0)
MODEL = ModelProfile(
    profile_id="m", provider="stub", model_name="stub", factor="large_general", pricing=PRICING
)
HARNESS = HarnessProfile(profile_id="generic", kind="generic", max_steps=8)


def _task(fixture_repo: Path) -> NavTask:
    return NavTask(
        task_id="t1",
        workspace=str(fixture_repo),
        prompt="Find the file where `class RateLimiter` is defined.",
        expected_files=["app/core/rate_limiter.py"],
    )


def _tool_msg(name: str, args: dict) -> ConversationMessage:
    return ConversationMessage(
        role="assistant",
        content=[TextBlock(text="..."), ToolUseBlock(name=name, input=args)],
    )


async def test_tool_loop_and_final_scoring(fixture_repo: Path) -> None:
    client = ScriptedClient(
        outputs=[
            _tool_msg("grep", {"pattern": "class RateLimiter"}),
            _tool_msg("read_file", {"path": "app/core/rate_limiter.py", "limit": 30}),
            "The relevant file is app/core/rate_limiter.py (sliding-window limiter).",
        ]
    )
    harness = GenericHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert outcome.success
    assert outcome.final_answer is not None
    assert outcome.final_answer["files"] == ["app/core/rate_limiter.py"]
    # The model received native tool schemas including distractors.
    tool_names = {t["name"] for t in client.calls[0].tools}
    assert {"grep", "read_file", "glob", "bash", "write_file"} <= tool_names


async def test_distractor_tool_returns_error_and_counts_retry(fixture_repo: Path) -> None:
    client = ScriptedClient(
        outputs=[
            _tool_msg("bash", {"command": "ls"}),
            _tool_msg("grep", {"pattern": "class RateLimiter"}),
            "Answer: app/core/rate_limiter.py",
        ]
    )
    harness = GenericHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert outcome.success
    assert outcome.retries == 1 and outcome.invalid_actions == 1
    bash_call = outcome.steps[0].tool_calls[0]
    assert bash_call.name == "bash" and bash_call.is_error


async def test_wrong_answer_fails(fixture_repo: Path) -> None:
    client = ScriptedClient(outputs=["The relevant file is app/auth/token.py."])
    harness = GenericHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert not outcome.success and outcome.error_type == "wrong_answer"


async def test_budget_exhaustion(fixture_repo: Path) -> None:
    client = ScriptedClient(
        outputs=[_tool_msg("glob", {"pattern": "**/*.py"}) for _ in range(20)]
    )
    harness = GenericHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert not outcome.success and outcome.error_type == "budget_exhausted"


async def test_oracle_stub_generic_solves_task(fixture_repo: Path) -> None:
    harness = GenericHarness(OracleStubClient(skill=1.0, seed=1), MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert outcome.success, outcome


def test_distractor_schemas_are_schema_only() -> None:
    names = [t["name"] for t in DISTRACTOR_TOOL_SCHEMAS]
    assert "bash" in names and "write_file" in names
    for schema in DISTRACTOR_TOOL_SCHEMAS:
        assert set(schema) == {"name", "description", "input_schema"}
