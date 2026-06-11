"""Deterministic verifier behavior."""

from __future__ import annotations

from pathlib import Path

from slm_harness.harness.state import CustomHarnessState, Hit
from slm_harness.schemas.actions import (
    AnswerAction,
    EscalateAction,
    EvidenceSpan,
    ReadAction,
    SearchAction,
)
from slm_harness.schemas.task import NavTask, VerificationSpec
from slm_harness.verifier import (
    extract_paths_from_text,
    score_final_answer,
    validate_action,
)


def _state(**kwargs) -> CustomHarnessState:
    state = CustomHarnessState(goal="find `RateLimiter`", max_steps=10)
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


def test_action_space_constraint(fixture_repo: Path) -> None:
    state = _state()
    # No files known yet: READ and ANSWER are invalid actions.
    res = validate_action(ReadAction(path="app/core/rate_limiter.py"), state, fixture_repo)
    assert not res.ok and res.retryable and "not valid now" in res.reason
    res = validate_action(AnswerAction(files=["app/core/rate_limiter.py"]), state, fixture_repo)
    assert not res.ok
    # ESCALATE is always allowed.
    res = validate_action(EscalateAction(reason="stuck"), state, fixture_repo)
    assert res.ok


def test_read_validation(fixture_repo: Path) -> None:
    state = _state(known_files=["app/core/rate_limiter.py"])
    assert validate_action(
        ReadAction(path="app/core/rate_limiter.py"), state, fixture_repo
    ).ok
    res = validate_action(ReadAction(path="app/core/missing.py"), state, fixture_repo)
    assert not res.ok and res.retryable and "does not exist" in res.reason
    res = validate_action(ReadAction(path="../../etc/passwd"), state, fixture_repo)
    assert not res.ok and "escapes workspace" in res.reason


def test_repeated_search_rejected(fixture_repo: Path) -> None:
    state = _state()
    from slm_harness.harness.state import SearchRecord

    state.searches.append(SearchRecord(pattern="RateLimiter", file_glob="**/*", n_hits=2))
    res = validate_action(SearchAction(pattern="RateLimiter"), state, fixture_repo)
    assert not res.ok and "already executed" in res.reason


def test_invalid_regex_rejected(fixture_repo: Path) -> None:
    res = validate_action(SearchAction(pattern="(["), _state(), fixture_repo)
    assert not res.ok and "invalid regex" in res.reason


def test_answer_citation_discipline(fixture_repo: Path) -> None:
    state = _state(known_files=["app/core/rate_limiter.py"])
    state.hits.append(Hit(path="app/core/rate_limiter.py", line=10, text="class RateLimiter"))
    from slm_harness.harness.state import ReadRecord

    state.reads.append(ReadRecord(path="app/core/rate_limiter.py", offset=0, limit=40, n_lines=28))
    # Citing an undiscovered (but existing) file is rejected.
    res = validate_action(AnswerAction(files=["app/auth/token.py"]), state, fixture_repo)
    assert not res.ok and "never discovered" in res.reason
    # Evidence span beyond EOF is rejected.
    res = validate_action(
        AnswerAction(
            files=["app/core/rate_limiter.py"],
            evidence=[EvidenceSpan(path="app/core/rate_limiter.py", line_start=1, line_end=9999)],
        ),
        state,
        fixture_repo,
    )
    assert not res.ok and "out of range" in res.reason
    # Valid answer passes.
    res = validate_action(
        AnswerAction(
            files=["app/core/rate_limiter.py"],
            evidence=[EvidenceSpan(path="app/core/rate_limiter.py", line_start=8, line_end=12)],
        ),
        state,
        fixture_repo,
    )
    assert res.ok


def test_score_final_answer(fixture_repo: Path) -> None:
    task = NavTask(
        task_id="t",
        workspace=str(fixture_repo),
        prompt="p",
        expected_files=["app/core/rate_limiter.py"],
    )
    assert score_final_answer(task, ["app/core/rate_limiter.py"], "", fixture_repo).ok
    assert not score_final_answer(task, ["app/auth/token.py"], "", fixture_repo).ok
    assert not score_final_answer(task, [], "", fixture_repo).ok
    # forbid_extra_files
    task.verification = VerificationSpec(forbid_extra_files=True)
    assert not score_final_answer(
        task, ["app/core/rate_limiter.py", "app/auth/token.py"], "", fixture_repo
    ).ok
    # regex method
    task.verification = VerificationSpec(method="answer_regex", answer_regex=r"sliding[- ]window")
    assert score_final_answer(task, [], "uses a sliding-window limiter", fixture_repo).ok
    assert not score_final_answer(task, [], "uses a token bucket", fixture_repo).ok


def test_extract_paths_from_text(fixture_repo: Path) -> None:
    text = (
        "The relevant files are:\n- app/core/rate_limiter.py\n- app/db/models.py.\n"
        "Also mentioned imaginary/file.py which does not exist."
    )
    paths = extract_paths_from_text(text, fixture_repo)
    assert "app/core/rate_limiter.py" in paths
    assert "app/db/models.py" in paths
    assert all("imaginary" not in p for p in paths)
