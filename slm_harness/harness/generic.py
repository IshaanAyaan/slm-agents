"""Generic harness adapter — the C1/C2/C4 baseline.

Reproduces a production general-purpose subagent faithfully:

- broad system prompt (capability text, workflow guidance, many tool descriptions),
- the OpenHarness general tool list with native tool calling, executed through the
  real ``ToolRegistry``/``BaseTool`` implementations,
- optional distractor tool schemas (write/bash/web/task tools advertised but not
  usable in this read-only role) to reproduce realistic prompt bloat,
- free-form conversation-history context accumulation,
- production-style tool-output truncation budgets
  (``openharness.services.tool_outputs`` defaults).

The final free-text answer is scored with the SAME deterministic scorer as the
custom harness (``verifier.score_final_answer``), so success criteria are identical
across conditions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from slm_harness import _bootstrap  # noqa: F401

from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
)
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry
from openharness.tools.file_read_tool import FileReadTool
from openharness.tools.glob_tool import GlobTool
from openharness.tools.grep_tool import GrepTool

from slm_harness.harness.state import HarnessOutcome
from slm_harness.model_clients.base import ModelCallRequest, SubagentModelClient
from slm_harness.schemas.config import HarnessProfile, ModelProfile
from slm_harness.schemas.runlog import StepLog, ToolCallLog, VerifierResult
from slm_harness.schemas.task import NavTask
from slm_harness.verifier import extract_paths_from_text, score_final_answer

GENERIC_SYSTEM_PROMPT = """You are a general-purpose AI subagent operating inside an \
agentic multi-agent system. You receive delegated tasks from an orchestrator agent and \
must complete them autonomously using the tools available to you.

# Capabilities
You can read files, search file contents with regular expressions, list files with glob \
patterns, and (in other deployments) edit files, run shell commands, fetch web pages, \
manage background tasks, and coordinate with other agents. For the current task you \
should rely on the read-only navigation tools.

# Workflow guidance
- Think carefully about the user's request and plan your approach before acting.
- Prefer targeted searches over reading entire files. Use grep with a regular \
expression to locate relevant code, then read_file to inspect the surrounding context.
- When you have located the relevant files and verified their contents, produce a \
final answer summarizing what you found. List every relevant file by its path relative \
to the working directory, and briefly justify each one.
- If a tool call fails, examine the error and adjust your approach. Do not give up \
after a single failure.
- Keep your responses concise. Do not include extraneous commentary.
- Never fabricate file paths or contents. Only cite files you have actually observed \
via tool results.
- When you are confident in your answer, reply WITHOUT any tool call. Your final \
message must contain the relevant file paths.

# Tool usage notes
- grep: searches file contents. Supports full regular expression syntax. Use the \
file_glob parameter to narrow by filename pattern, e.g. "**/*.py".
- read_file: reads a UTF-8 text file with line numbers, supports offset/limit \
pagination for large files.
- glob: lists files matching a glob pattern relative to the working directory.
- Other tools advertised in your tool list may be unavailable in this deployment; \
if a tool returns an unavailability error, fall back to the navigation tools above.

# Environment
You are operating in a read-only checkout of a software repository. The working \
directory is the repository root. Paths in your answer should be relative to it.
"""

# Schema-only distractor tools: advertised to the model (prompt bloat, realistic
# action-selection burden) but not executable in this read-only subagent role.
DISTRACTOR_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": name,
        "description": desc,
        "input_schema": {"type": "object", "properties": props, "required": req},
    }
    for name, desc, props, req in [
        (
            "bash",
            "Run a shell command in the workspace and return stdout/stderr. Supports a "
            "timeout parameter and background execution for long-running commands.",
            {"command": {"type": "string"}, "timeout": {"type": "number"}},
            ["command"],
        ),
        (
            "write_file",
            "Create or overwrite a file with the given contents. Parent directories are "
            "created automatically.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        (
            "edit_file",
            "Apply an exact string replacement edit to a file. The old string must be "
            "unique within the file.",
            {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            ["path", "old_string", "new_string"],
        ),
        (
            "web_search",
            "Search the web and return a list of result snippets with URLs.",
            {"query": {"type": "string"}},
            ["query"],
        ),
        (
            "web_fetch",
            "Fetch a URL and return the page content as markdown.",
            {"url": {"type": "string"}},
            ["url"],
        ),
        (
            "task_create",
            "Create a background task tracked by the task manager.",
            {"subject": {"type": "string"}, "description": {"type": "string"}},
            ["subject"],
        ),
        (
            "send_message",
            "Send a message to another agent on the team via the mailbox.",
            {"recipient": {"type": "string"}, "message": {"type": "string"}},
            ["recipient", "message"],
        ),
    ]
]


def build_generic_registry(allowed_tools: list[str]) -> ToolRegistry:
    """Real, executable read-only tools from the OpenHarness tool set."""
    registry = ToolRegistry()
    available: list[BaseTool] = [FileReadTool(), GrepTool(), GlobTool()]
    for tool in available:
        if tool.name in allowed_tools:
            registry.register(tool)
    return registry


class GenericHarness:
    """Baseline harness: broad prompt, broad tools, free-form context."""

    def __init__(
        self,
        client: SubagentModelClient,
        model_profile: ModelProfile,
        harness_profile: HarnessProfile,
    ) -> None:
        self._client = client
        self._model = model_profile
        self._profile = harness_profile

    async def run(self, task: NavTask) -> HarnessOutcome:
        """Run the production-style tool loop until a text-only answer or budget."""
        workspace = Path(task.workspace).resolve()
        registry = build_generic_registry(task.allowed_tools)
        tools_schema = registry.to_api_schema()
        if self._profile.include_distractor_tools:
            tools_schema = tools_schema + DISTRACTOR_TOOL_SCHEMAS
        ctx = ToolExecutionContext(cwd=workspace)
        outcome = HarnessOutcome(success=False)
        started = time.monotonic()

        messages: list[ConversationMessage] = [
            ConversationMessage.from_user_text(
                f"{task.prompt}\n\nWorking directory: repository root. "
                "Report file paths relative to it."
            )
        ]

        for turn in range(self._profile.max_steps):
            step = StepLog(
                step_index=turn,
                observation=self._render_transcript(messages),
            )
            t0 = time.monotonic()
            try:
                response = await self._client.complete(
                    ModelCallRequest(
                        model=self._model.model_name,
                        system_prompt=GENERIC_SYSTEM_PROMPT,
                        messages=messages,
                        max_tokens=self._model.max_tokens,
                        tools=tools_schema,
                    )
                )
            except Exception as exc:
                step.latency_s = time.monotonic() - t0
                step.action_valid = False
                step.verifier = VerifierResult(ok=False, reason=f"model error: {exc}")
                outcome.steps.append(step)
                outcome.error_type = "model_error"
                outcome.latency_s = time.monotonic() - started
                return outcome
            step.latency_s = time.monotonic() - t0
            step.model_output = response.text
            step.input_tokens = response.usage.input_tokens
            step.output_tokens = response.usage.output_tokens
            outcome.input_tokens += response.usage.input_tokens
            outcome.output_tokens += response.usage.output_tokens

            tool_uses = response.message.tool_uses
            if not tool_uses:
                # Final free-text answer: score with the shared deterministic scorer.
                files = extract_paths_from_text(response.text, workspace)
                final = score_final_answer(task, files, response.text, workspace)
                step.parsed_action = {"action": "FINAL_ANSWER", "files": files}
                step.verifier = final
                outcome.steps.append(step)
                outcome.final_answer = {"files": files, "text": response.text[:2000]}
                outcome.final_verifier = final.model_dump()
                outcome.success = final.ok
                outcome.error_type = "" if final.ok else "wrong_answer"
                outcome.latency_s = time.monotonic() - started
                return outcome

            messages.append(response.message)
            result_blocks: list[ToolResultBlock] = []
            had_error = False
            for tu in tool_uses:
                output, is_error = await self._run_tool(registry, tu.name, tu.input, ctx)
                output = self._truncate(output)
                step.tool_calls.append(
                    ToolCallLog(
                        name=tu.name,
                        input=dict(tu.input),
                        output_chars=len(output),
                        is_error=is_error,
                    )
                )
                if is_error:
                    had_error = True
                result_blocks.append(
                    ToolResultBlock(tool_use_id=tu.id, content=output, is_error=is_error)
                )
            if had_error:
                # In the generic harness an error costs a whole extra loop turn.
                outcome.retries += 1
                outcome.invalid_actions += 1
                step.action_valid = False
            outcome.steps.append(step)
            messages.append(ConversationMessage(role="user", content=list(result_blocks)))

        outcome.error_type = "budget_exhausted"
        outcome.latency_s = time.monotonic() - started
        return outcome

    async def _run_tool(
        self,
        registry: ToolRegistry,
        name: str,
        raw_input: dict[str, Any],
        ctx: ToolExecutionContext,
    ) -> tuple[str, bool]:
        """Execute a tool call; distractor/unknown tools return an error result."""
        tool = registry.get(name)
        if tool is None:
            return (
                f"Tool '{name}' is not available in this subagent deployment. "
                "Use grep/read_file/glob.",
                True,
            )
        try:
            args = tool.input_model.model_validate(raw_input)
        except Exception as exc:
            return f"Invalid arguments for {name}: {exc}", True
        result = await tool.execute(args, ctx)
        return result.output, result.is_error

    def _truncate(self, output: str) -> str:
        """Mirror production inline tool-output budgets."""
        budget = self._profile.tool_output_char_budget
        if len(output) <= budget:
            return output
        return output[:budget] + f"\n… [truncated {len(output) - budget} chars]"

    @staticmethod
    def _render_transcript(messages: list[ConversationMessage]) -> str:
        """Compact transcript snapshot stored in logs (not shown to the model)."""
        parts: list[str] = []
        for msg in messages[-4:]:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(f"{msg.role}: {block.text[:400]}")
                elif isinstance(block, ToolResultBlock):
                    parts.append(f"tool_result: {block.content[:200]}")
        return "\n".join(parts)[:2400]
