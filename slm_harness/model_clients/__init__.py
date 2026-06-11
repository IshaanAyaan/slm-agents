"""Model-call abstraction: one protocol, real adapters, and offline stub clients."""

from slm_harness.model_clients.base import ModelCallRequest, ModelCallResponse, SubagentModelClient
from slm_harness.model_clients.stub import OracleStubClient, ScriptedClient

__all__ = [
    "ModelCallRequest",
    "ModelCallResponse",
    "OracleStubClient",
    "ScriptedClient",
    "SubagentModelClient",
]
