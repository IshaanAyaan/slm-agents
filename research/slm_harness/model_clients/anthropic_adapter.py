"""Real-model adapter wrapping the production OpenHarness Anthropic client.

Used for large-general-model conditions (C1/C6) in real runs. Requires an API key in
the environment variable named by the model profile (never hardcoded).
"""

from __future__ import annotations

import os

from research.slm_harness import _bootstrap  # noqa: F401

from openharness.api.client import (
    AnthropicApiClient,
    ApiMessageCompleteEvent,
    ApiMessageRequest,
)

from research.slm_harness.model_clients.base import ModelCallRequest, ModelCallResponse
from research.slm_harness.schemas.config import ModelProfile


class AnthropicAdapter:
    """SubagentModelClient backed by openharness.api.client.AnthropicApiClient."""

    def __init__(self, profile: ModelProfile) -> None:
        env_name = profile.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_name, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set ${env_name} to run real {profile.model_name} calls. "
                "Use the stub provider for offline smoke runs."
            )
        self._client = AnthropicApiClient(api_key=api_key, base_url=profile.base_url)

    async def complete(self, request: ModelCallRequest) -> ModelCallResponse:
        """Run one non-interactive completion via the streaming production client."""
        api_request = ApiMessageRequest(
            model=request.model,
            messages=request.messages,
            system_prompt=request.system_prompt or None,
            max_tokens=request.max_tokens,
            tools=request.tools,
        )
        final: ApiMessageCompleteEvent | None = None
        async for event in self._client.stream_message(api_request):
            if isinstance(event, ApiMessageCompleteEvent):
                final = event
        if final is None:
            raise RuntimeError("stream ended without a completed message")
        return ModelCallResponse(
            message=final.message, usage=final.usage, stop_reason=final.stop_reason
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
