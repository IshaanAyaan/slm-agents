"""Provider-agnostic model call protocol.

Reuses the OpenHarness message and usage models so real adapters can wrap the
production provider abstraction directly, while stub clients keep the whole
pipeline runnable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from slm_harness import _bootstrap  # noqa: F401

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage


@dataclass(frozen=True)
class ModelCallRequest:
    """One model invocation.

    - Generic harness: full conversation history + native tool schemas.
    - Custom harness: short system prompt + a single user message holding the
      JSON observation; ``tools`` is empty and the reply must be a JSON action.
    """

    model: str
    messages: list[ConversationMessage]
    system_prompt: str = ""
    max_tokens: int = 1024
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCallResponse:
    """Completed model response."""

    message: ConversationMessage
    usage: UsageSnapshot
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        """Concatenated text content."""
        return self.message.text


@runtime_checkable
class SubagentModelClient(Protocol):
    """Anything that can complete a ModelCallRequest."""

    async def complete(self, request: ModelCallRequest) -> ModelCallResponse:
        """Run one model call and return the final message + usage."""
        ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) used only by stub clients."""
    return max(1, len(text) // 4)
