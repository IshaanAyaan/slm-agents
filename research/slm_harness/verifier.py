"""Deterministic verification for the file/code navigation subagent role.

Three layers, all cheap and rule-based (no model calls):

1. ``validate_action`` — pre-execution: the parsed action must be in the currently
   valid action set, reference real in-workspace paths, and use a compilable regex.
2. ``verify_step_result`` — post-execution: the environment call must have produced
   a well-formed result.
3. ``score_final_answer`` — terminal: the answer must cite existing files, evidence
   spans must exist within those files, and the task's success criteria
   (expected files / answer regex) must hold. Shared by BOTH harnesses so final
   scoring is identical across conditions.
"""

from __future__ import annotations

import re
from pathlib import Path

from research.slm_harness.harness.state import CustomHarnessState
from research.slm_harness.schemas.actions import (
    AnswerAction,
    EscalateAction,
    ReadAction,
    SearchAction,
    SubagentAction,
)
from research.slm_harness.schemas.runlog import VerifierResult
from research.slm_harness.schemas.task import NavTask


def resolve_in_workspace(workspace: Path, candidate: str) -> Path | None:
    """Resolve a path inside the workspace; None if it escapes the root."""
    try:
        resolved = (workspace / candidate).resolve()
        resolved.relative_to(workspace.resolve())
    except (ValueError, OSError):
        return None
    return resolved


def validate_action(
    action: SubagentAction, state: CustomHarnessState, workspace: Path
) -> VerifierResult:
    """Pre-execution validation (detects invalid action attempts)."""
    checks: dict[str, bool] = {}
    name = action.action

    checks["action_in_valid_set"] = name in state.valid_actions()
    if not checks["action_in_valid_set"]:
        return VerifierResult(
            ok=False,
            reason=f"action {name} is not valid now; valid actions: {state.valid_actions()}",
            retryable=True,
            checks=checks,
        )

    if isinstance(action, SearchAction):
        try:
            re.compile(action.pattern)
            checks["pattern_compiles"] = True
        except re.error as exc:
            checks["pattern_compiles"] = False
            return VerifierResult(
                ok=False, reason=f"invalid regex: {exc}", retryable=True, checks=checks
            )
        repeated = any(
            s.pattern == action.pattern and s.file_glob == action.file_glob
            for s in state.searches
        )
        checks["not_repeated_search"] = not repeated
        if repeated:
            return VerifierResult(
                ok=False,
                reason="identical search already executed; vary the pattern or glob",
                retryable=True,
                checks=checks,
            )

    elif isinstance(action, ReadAction):
        resolved = resolve_in_workspace(workspace, action.path)
        checks["path_in_workspace"] = resolved is not None
        if resolved is None:
            return VerifierResult(
                ok=False, reason=f"path escapes workspace: {action.path}", retryable=True, checks=checks
            )
        checks["path_exists"] = resolved.is_file()
        if not resolved.is_file():
            return VerifierResult(
                ok=False,
                reason=f"file does not exist: {action.path} (known files: {state.known_files[:5]})",
                retryable=True,
                checks=checks,
            )

    elif isinstance(action, AnswerAction):
        for f in action.files:
            resolved = resolve_in_workspace(workspace, f)
            if resolved is None or not resolved.is_file():
                checks["answer_files_exist"] = False
                return VerifierResult(
                    ok=False, reason=f"answer cites nonexistent file: {f}", retryable=True, checks=checks
                )
        checks["answer_files_exist"] = True
        read_paths = {r.path for r in state.reads}
        uncited = [f for f in action.files if f not in read_paths and f not in state.known_files]
        checks["answer_files_discovered"] = not uncited
        if uncited:
            return VerifierResult(
                ok=False,
                reason=f"answer cites files never discovered via SEARCH/READ: {uncited}",
                retryable=True,
                checks=checks,
            )
        for span in action.evidence:
            resolved = resolve_in_workspace(workspace, span.path)
            if resolved is None or not resolved.is_file():
                checks["evidence_paths_exist"] = False
                return VerifierResult(
                    ok=False, reason=f"evidence cites nonexistent file: {span.path}", retryable=True, checks=checks
                )
            n_lines = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
            if span.line_end < span.line_start or span.line_end > n_lines:
                checks["evidence_spans_valid"] = False
                return VerifierResult(
                    ok=False,
                    reason=(
                        f"evidence span {span.path}:{span.line_start}-{span.line_end} "
                        f"out of range (file has {n_lines} lines)"
                    ),
                    retryable=True,
                    checks=checks,
                )
        checks.setdefault("evidence_paths_exist", True)
        checks.setdefault("evidence_spans_valid", True)

    elif isinstance(action, EscalateAction):
        checks["escalate_allowed"] = True

    return VerifierResult(ok=True, checks=checks)


def verify_step_result(action: SubagentAction, output: str, is_error: bool) -> VerifierResult:
    """Post-execution check on the environment call."""
    if is_error:
        return VerifierResult(
            ok=False, reason=f"tool error during {action.action}: {output[:200]}", retryable=True
        )
    return VerifierResult(ok=True)


_PATH_TOKEN_RE = re.compile(r"[\w./\-]+\.\w{1,8}")


def extract_paths_from_text(text: str, workspace: Path) -> list[str]:
    """Leniently extract workspace-relative file paths from free-form text.

    Used to score the GENERIC harness's final answer with the same criteria as
    the custom harness's structured ANSWER.
    """
    found: list[str] = []
    ws = workspace.resolve()
    for token in _PATH_TOKEN_RE.findall(text):
        candidate = token.strip(".,;:")
        rel = candidate
        if candidate.startswith(str(ws)):
            rel = str(Path(candidate).relative_to(ws))
        resolved = resolve_in_workspace(workspace, rel)
        if resolved is not None and resolved.is_file():
            posix = Path(rel).as_posix()
            if posix not in found:
                found.append(posix)
    return found


def score_final_answer(
    task: NavTask, files: list[str], answer_text: str, workspace: Path
) -> VerifierResult:
    """Terminal task scoring, identical across all conditions."""
    checks: dict[str, bool] = {}
    spec = task.verification
    norm = [Path(f).as_posix() for f in files]

    files_ok = True
    if spec.method in ("expected_files", "both"):
        expected = [Path(f).as_posix() for f in task.expected_files]
        if spec.require_all_files:
            files_ok = all(e in norm for e in expected)
        else:
            files_ok = any(e in norm for e in expected)
        if spec.forbid_extra_files and files_ok:
            files_ok = all(f in expected for f in norm)
        checks["expected_files"] = files_ok

    regex_ok = True
    if spec.method in ("answer_regex", "both"):
        if spec.answer_regex is None:
            regex_ok = False
        else:
            regex_ok = re.search(spec.answer_regex, answer_text) is not None
        checks["answer_regex"] = regex_ok

    ok = files_ok and regex_ok
    reason = "" if ok else (
        f"expected files {task.expected_files}, got {norm}"
        if not files_ok
        else f"answer text did not match /{spec.answer_regex}/"
    )
    return VerifierResult(ok=ok, reason=reason, retryable=False, checks=checks)
