"""MCP-ready integration layer.

A frontier coding agent (Claude Code, Codex, ...) calls a few coarse MCP tools;
the harness behind them owns schema validation, deployment-time guards, retry,
rules fallback, and routing to the right small specialist. The specialists stay
invisible to the orchestrator: it sees verified structured outputs only.
"""
