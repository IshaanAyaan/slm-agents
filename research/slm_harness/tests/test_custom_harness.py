"""Custom harness: state transitions, retry localization, full task runs."""

from __future__ import annotations

import json
from pathlib import Path

from research.slm_harness.harness.custom import CustomSearchHarness
from research.slm_harness.harness.state import CustomHarnessState
from research.slm_harness.model_clients.stub import OracleStubClient, ScriptedClient
from research.slm_harness.schemas.config import HarnessProfile, ModelProfile, PricingConfig
from research.slm_harness.schemas.task import NavTask

PRICING = PricingConfig(usd_per_million_input=1.0, usd_per_million_output=2.0)
MODEL = ModelProfile(
    profile_id="m", provider="stub", model_name="stub", factor="small_general", pricing=PRICING
)
HARNESS = HarnessProfile(profile_id="custom", kind="custom", max_steps=8, max_retries_per_step=2)


def _task(fixture_repo: Path) -> NavTask:
    return NavTask(
        task_id="t1",
        workspace=str(fixture_repo),
        prompt="Find the file where `class RateLimiter` is defined.",
        expected_files=["app/core/rate_limiter.py"],
    )


def test_state_transitions() -> None:
    state = CustomHarnessState(goal="g", max_steps=5)
    assert state.valid_actions() == ["SEARCH", "ESCALATE"]
    from research.slm_harness.harness.state import Hit

    state.record_search("x", "**/*", [Hit(path="a.py", line=3, text="x = 1")])
    assert state.known_files == ["a.py"]
    assert "READ" in state.valid_actions() and "ANSWER" not in state.valid_actions()
    state.record_read("a.py", 0, 40, "      1\tx = 1", 1)
    assert "ANSWER" in state.valid_actions()
    obs = json.loads(state.build_observation())
    assert obs["goal"] == "g" and obs["files"] == ["a.py"]
    assert obs["last_result"]["type"] == "read"


async def test_successful_trajectory(fixture_repo: Path) -> None:
    client = ScriptedClient(
        outputs=[
            '{"action":"SEARCH","pattern":"class RateLimiter"}',
            json.dumps(
                {"action": "READ", "path": "app/core/rate_limiter.py", "offset": 6, "limit": 20}
            ),
            json.dumps(
                {
                    "action": "ANSWER",
                    "files": ["app/core/rate_limiter.py"],
                    "evidence": [
                        {"path": "app/core/rate_limiter.py", "line_start": 7, "line_end": 12}
                    ],
                    "confidence": 0.95,
                }
            ),
        ]
    )
    harness = CustomSearchHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert outcome.success and outcome.error_type == ""
    assert outcome.retries == 0 and outcome.invalid_actions == 0
    # The model only ever saw compact JSON observations.
    for call in client.calls:
        json.loads(call.messages[-1].text)


async def test_retry_localization(fixture_repo: Path) -> None:
    """An invalid step is re-prompted with feedback; the task does NOT restart."""
    client = ScriptedClient(
        outputs=[
            "I will look around the repo first.",  # invalid: prose
            '{"action":"SEARCH","pattern":"class RateLimiter"}',  # retry of step 0
            '{"action":"READ","path":"app/missing.py"}',  # invalid: ghost file
            '{"action":"READ","path":"app/core/rate_limiter.py","offset":0,"limit":30}',
            json.dumps(
                {
                    "action": "ANSWER",
                    "files": ["app/core/rate_limiter.py"],
                    "evidence": [
                        {"path": "app/core/rate_limiter.py", "line_start": 1, "line_end": 5}
                    ],
                }
            ),
        ]
    )
    harness = CustomSearchHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert outcome.success
    assert outcome.invalid_actions == 2 and outcome.retries == 2
    # Retried calls carry verifier feedback in the observation; the goal is unchanged.
    retry_obs = json.loads(client.calls[1].messages[-1].text)
    assert "feedback" in retry_obs and "JSON" in retry_obs["feedback"]
    ghost_retry_obs = json.loads(client.calls[3].messages[-1].text)
    assert "does not exist" in ghost_retry_obs["feedback"]
    # Step indices show localization: search retried at step 0, read retried at step 1.
    assert [s.step_index for s in outcome.steps] == [0, 0, 1, 1, 2]
    assert [s.retry_index for s in outcome.steps] == [0, 1, 0, 1, 0]


async def test_terminal_failure_after_retry_budget(fixture_repo: Path) -> None:
    client = ScriptedClient(outputs=["nope", "still nope", "absolutely not"])
    harness = CustomSearchHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert not outcome.success and outcome.error_type == "invalid_action"
    assert len(outcome.steps) == 3  # max_retries_per_step + 1


async def test_escalation(fixture_repo: Path) -> None:
    client = ScriptedClient(outputs=['{"action":"ESCALATE","reason":"cannot find it"}'])
    harness = CustomSearchHarness(client, MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert not outcome.success and outcome.escalated and outcome.error_type == "escalated"


async def test_oracle_stub_solves_smoke_task(fixture_repo: Path) -> None:
    harness = CustomSearchHarness(OracleStubClient(skill=1.0, seed=0), MODEL, HARNESS)
    outcome = await harness.run(_task(fixture_repo))
    assert outcome.success, outcome
    assert outcome.input_tokens > 0 and outcome.output_tokens > 0
