"""Phase-2 developer-agent subroutines for sub-500M specialist models.

Each subroutine is a narrow, schema-bound, deterministically-verifiable task that a
coding-agent harness can delegate to a small language model. The harness owns state,
tool access, retrieval, verification, retries, and fallback; the model only maps a
rendered observation to a structured output.
"""
