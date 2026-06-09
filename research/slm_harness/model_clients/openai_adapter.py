"""OpenAI-compatible adapter for self-hosted SLMs (vLLM, llama.cpp server, TGI).

Used for small-model conditions (C2/C3/C4/C5) in real runs, pointing at a local
endpoint such as vLLM serving Qwen3-4B (optionally with a LoRA adapter loaded).
No credentials are hardcoded; the API key (if any) comes from the environment
variable named in the model profile.
"""

from __future__ import annotations

import json
import os
from typing import Any

from research.slm_harness import _bootstrap  # noqa: F401

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from research.slm_harness.model_clients.base import ModelCallRequest, ModelCallResponse
from research.slm_harness.schemas.config import ModelProfile


def _to_openai_messages(request: ModelCallRequest) -> list[dict[str, Any]]:
    """Convert OpenHarness conversation messages to OpenAI chat format."""
    out: list[dict[str, Any]] = []
    if request.system_prompt:
        out.append({"role": "system", "content": request.system_prompt})
    for msg in request.messages:
        tool_calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]
        tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
        text = msg.text
        if msg.role == "assistant" and tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc.id[:40],
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
        elif tool_results:
            for tr in tool_results:
                out.append(
                    {"role": "tool", "tool_call_id": tr.tool_use_id[:40], "content": tr.content}
                )
            if text:
                out.append({"role": "user", "content": text})
        else:
            out.append({"role": msg.role, "content": text})
    return out


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool schemas to OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools
    ]


class OpenAICompatAdapter:
    """SubagentModelClient for OpenAI-compatible endpoints (vLLM et al.)."""

    def __init__(self, profile: ModelProfile) -> None:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai to use OpenAICompatAdapter") from exc
        api_key = os.environ.get(profile.api_key_env or "", "") or "EMPTY"
        if not profile.base_url:
            raise RuntimeError(
                f"Model profile {profile.profile_id} needs base_url (e.g. http://localhost:8000/v1)"
            )
        self._client = AsyncOpenAI(api_key=api_key, base_url=profile.base_url)
        self._profile = profile

    async def complete(self, request: ModelCallRequest) -> ModelCallResponse:
        """Run one chat completion, converting tool calls both ways."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": _to_openai_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            kwargs["tools"] = _to_openai_tools(request.tools)
        if self._profile.adapter_path:
            # vLLM serves LoRA adapters as separate model names; callers may also
            # set model_name directly to the served adapter id.
            kwargs["model"] = request.metadata.get("served_model", request.model)
        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        blocks: list[TextBlock | ToolUseBlock] = []
        if choice.message.content:
            blocks.append(TextBlock(text=choice.message.content))
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            blocks.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))
        usage = completion.usage
        return ModelCallResponse(
            message=ConversationMessage(role="assistant", content=blocks),
            usage=UsageSnapshot(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            stop_reason=choice.finish_reason,
        )
