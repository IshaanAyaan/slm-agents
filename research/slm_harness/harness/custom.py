"""Custom (co-designed) harness for the file/code search & navigation subagent role.

Implements the five harness requirements:

1. Constrained action space — only currently-valid actions are exposed
   (``CustomHarnessState.valid_actions``) and validated pre-execution.
2. Minimal structured observation — compact JSON, no transcript accumulation.
3. Externalized state — goal/history/files/failures live in ``CustomHarnessState``.
4. Cheap deterministic verification after every step (``verifier``).
5. Retry localization — on a retryable failure only the failed step is re-prompted
   with verifier feedback; the task never restarts.

SEARCH/READ delegate to the *same* OpenHarness ``GrepTool``/``FileReadTool``
implementations used by the generic harness, so the environment is held constant
across conditions and only the model-facing interface varies.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from research.slm_harness import _bootstrap  # noqa: F401

from openharness.engine.messages import ConversationMessage
from openharness.tools.base import ToolExecutionContext
from openharness.tools.file_read_tool import FileReadTool, FileReadToolInput
from openharness.tools.grep_tool import GrepTool, GrepToolInput

from research.slm_harness.harness.state import CustomHarnessState, HarnessOutcome, Hit
from research.slm_harness.model_clients.base import ModelCallRequest, SubagentModelClient
from research.slm_harness.schemas.actions import (
    AnswerAction,
    EscalateAction,
    ReadAction,
    SearchAction,
    SubagentAction,
    parse_action,
)
from research.slm_harness.schemas.config import HarnessProfile, ModelProfile
from research.slm_harness.schemas.runlog import StepLog, ToolCallLog, VerifierResult
from research.slm_harness.schemas.task import NavTask
from research.slm_harness.verifier import score_final_answer, validate_action, verify_step_result

CUSTOM_SYSTEM_PROMPT = (
    "You are a file-search subagent. Each turn you get one JSON observation with the "
    "goal, your past actions, and the currently valid actions. Reply with EXACTLY one "
    "JSON object and nothing else. Actions:\n"
    '{"action":"SEARCH","pattern":"<regex>","file_glob":"**/*","root":"."}\n'
    '{"action":"READ","path":"<file>","offset":0,"limit":80}\n'
    '{"action":"ANSWER","files":["<file>"],"evidence":[{"path":"<file>",'
    '"line_start":1,"line_end":2}],"answer_text":"","confidence":0.9}\n'
    '{"action":"ESCALATE","reason":"<why>"}\n'
    "Only use actions listed in valid_actions. Cite only files you have discovered."
)

_GREP_LINE_RE = re.compile(r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<text>.*)$")


class CustomSearchHarness:
    """Narrow harness: tiny action set, external state, per-step verification."""

    def __init__(
        self,
        client: SubagentModelClient,
        model_profile: ModelProfile,
        harness_profile: HarnessProfile,
    ) -> None:
        self._client = client
        self._model = model_profile
        self._profile = harness_profile
        self._grep = GrepTool()
        self._read = FileReadTool()

    async def run(self, task: NavTask) -> HarnessOutcome:
        """Run the task to completion (answer, escalation, or budget exhaustion)."""
        workspace = Path(task.workspace).resolve()
        state = CustomHarnessState(goal=task.prompt, max_steps=self._profile.max_steps)
        outcome = HarnessOutcome(success=False)
        ctx = ToolExecutionContext(cwd=workspace)
        started = time.monotonic()

        while state.step < self._profile.max_steps:
            action, step_logs = await self._one_step_with_retries(task, state, workspace, ctx)
            outcome.steps.extend(step_logs)
            for log_entry in step_logs:
                outcome.input_tokens += log_entry.input_tokens
                outcome.output_tokens += log_entry.output_tokens
                if not log_entry.action_valid:
                    outcome.invalid_actions += 1
                if log_entry.verifier is not None and not log_entry.verifier.ok:
                    outcome.verifier_failures += 1
                if log_entry.retry_index > 0:
                    outcome.retries += 1

            if action is None:
                # Step failed even after localized retries -> terminal failure.
                outcome.error_type = "invalid_action"
                outcome.latency_s = time.monotonic() - started
                return outcome

            if isinstance(action, EscalateAction):
                outcome.escalated = True
                outcome.error_type = "escalated"
                outcome.final_answer = {"action": "ESCALATE", "reason": action.reason}
                outcome.latency_s = time.monotonic() - started
                return outcome

            if isinstance(action, AnswerAction):
                final = score_final_answer(task, action.files, action.answer_text, workspace)
                outcome.final_answer = action.model_dump()
                outcome.final_verifier = final.model_dump()
                outcome.success = final.ok
                outcome.error_type = "" if final.ok else "wrong_answer"
                outcome.latency_s = time.monotonic() - started
                return outcome

            state.step += 1

        outcome.error_type = "budget_exhausted"
        outcome.latency_s = time.monotonic() - started
        return outcome

    # ------------------------------------------------------------------

    async def _one_step_with_retries(
        self,
        task: NavTask,
        state: CustomHarnessState,
        workspace: Path,
        ctx: ToolExecutionContext,
    ) -> tuple[SubagentAction | None, list[StepLog]]:
        """Run one logical step; on retryable failure re-prompt ONLY this step."""
        logs: list[StepLog] = []
        for retry in range(self._profile.max_retries_per_step + 1):
            observation = state.build_observation(self._profile.observation_char_budget)
            log_entry = StepLog(step_index=state.step, retry_index=retry, observation=observation)
            t0 = time.monotonic()
            try:
                response = await self._client.complete(
                    ModelCallRequest(
                        model=self._model.model_name,
                        system_prompt=CUSTOM_SYSTEM_PROMPT,
                        messages=[ConversationMessage.from_user_text(observation)],
                        max_tokens=self._model.max_tokens,
                    )
                )
            except Exception as exc:  # provider failure
                log_entry.action_valid = False
                log_entry.verifier = VerifierResult(
                    ok=False, reason=f"model error: {exc}", retryable=True
                )
                log_entry.latency_s = time.monotonic() - t0
                logs.append(log_entry)
                state.record_failure(f"model error: {exc}")
                continue
            log_entry.latency_s = time.monotonic() - t0
            log_entry.model_output = response.text
            log_entry.input_tokens = response.usage.input_tokens
            log_entry.output_tokens = response.usage.output_tokens

            action, parse_err = parse_action(response.text)
            if action is None:
                log_entry.action_valid = False
                log_entry.verifier = VerifierResult(ok=False, reason=parse_err, retryable=True)
                logs.append(log_entry)
                state.record_failure(f"invalid action: {parse_err}")
                continue
            log_entry.parsed_action = action.model_dump()

            pre = validate_action(action, state, workspace)
            if not pre.ok:
                log_entry.action_valid = False
                log_entry.verifier = pre
                logs.append(log_entry)
                state.record_failure(pre.reason)
                if not pre.retryable:
                    return None, logs
                continue

            # Terminal actions are executed by the caller.
            if isinstance(action, (AnswerAction, EscalateAction)):
                log_entry.verifier = pre
                logs.append(log_entry)
                state.clear_feedback()
                return action, logs

            # Execute the environment call.
            tool_log, output, is_error = await self._execute(action, workspace, ctx)
            log_entry.tool_calls.append(tool_log)
            post = verify_step_result(action, output, is_error)
            log_entry.verifier = post
            logs.append(log_entry)
            if not post.ok:
                state.record_failure(post.reason)
                continue

            self._fold_result(action, output, state, workspace)
            state.clear_feedback()
            return action, logs

        return None, logs

    # ------------------------------------------------------------------

    async def _execute(
        self, action: SubagentAction, workspace: Path, ctx: ToolExecutionContext
    ) -> tuple[ToolCallLog, str, bool]:
        """Execute SEARCH/READ via the shared OpenHarness tools."""
        return await execute_env_action(action, ctx, grep=self._grep, read=self._read)

    def _fold_result(
        self, action: SubagentAction, output: str, state: CustomHarnessState, workspace: Path
    ) -> None:
        """Convert raw tool output into minimal structured state (requirement 2/3)."""
        fold_result_into_state(action, output, state, workspace)


# ----------------------------------------------------------------------
# Module-level executors, shared with the distillation replay (distill.py).
# ----------------------------------------------------------------------


async def execute_env_action(
    action: SubagentAction,
    ctx: ToolExecutionContext,
    *,
    grep: GrepTool | None = None,
    read: FileReadTool | None = None,
) -> tuple[ToolCallLog, str, bool]:
    """Execute SEARCH/READ via the shared OpenHarness tools."""
    if isinstance(action, SearchAction):
        args = GrepToolInput(
            pattern=action.pattern,
            root=action.root if action.root != "." else None,
            file_glob=action.file_glob,
            limit=50,
        )
        result = await (grep or GrepTool()).execute(args, ctx)
        return (
            ToolCallLog(
                name="grep",
                input=args.model_dump(),
                output_chars=len(result.output),
                is_error=result.is_error,
            ),
            result.output,
            result.is_error,
        )
    if isinstance(action, ReadAction):
        read_args = FileReadToolInput(path=action.path, offset=action.offset, limit=action.limit)
        result = await (read or FileReadTool()).execute(read_args, ctx)
        return (
            ToolCallLog(
                name="read_file",
                input=read_args.model_dump(),
                output_chars=len(result.output),
                is_error=result.is_error,
            ),
            result.output,
            result.is_error,
        )
    raise ValueError(f"not an environment action: {action.action}")


def fold_result_into_state(
    action: SubagentAction, output: str, state: CustomHarnessState, workspace: Path
) -> None:
    """Convert raw tool output into minimal structured state (requirement 2/3)."""
    if isinstance(action, SearchAction):
        hits: list[Hit] = []
        if output.strip() != "(no matches)":
            for line in output.splitlines():
                m = _GREP_LINE_RE.match(line)
                if m is None:
                    continue
                raw_path = m.group("path")
                p = Path(raw_path)
                if p.is_absolute():
                    try:
                        raw_path = str(p.relative_to(workspace.resolve()))
                    except ValueError:
                        continue
                rel = Path(raw_path).as_posix().lstrip("./")
                hits.append(Hit(path=rel, line=int(m.group("line")), text=m.group("text")))
        state.record_search(action.pattern, action.file_glob, hits)
    elif isinstance(action, ReadAction):
        n_lines = len([ln for ln in output.splitlines() if ln.strip()])
        state.record_read(
            Path(action.path).as_posix(), action.offset, action.limit, output, n_lines
        )
