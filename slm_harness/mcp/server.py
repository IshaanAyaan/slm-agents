"""Minimal stdio MCP server exposing the specialist harness.

Stdlib-only JSON-RPC 2.0 over stdin/stdout implementing the MCP subset that
coding agents need: initialize, tools/list, tools/call. Register with e.g.:

    claude mcp add slm-specialists -- python -m slm_harness.mcp.server

Backend selection via env: SLM_BACKEND=rules|stub|hf (default rules; hf needs
SLM_MODELS_ROOT and optionally SLM_SIZE).
"""

from __future__ import annotations

import json
import sys

from slm_harness.mcp.tools import TOOL_DEFS, call_tool, make_backend

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "slm-specialists", "version": "1.0.0"}


def handle(request: dict, backend) -> dict | None:
    """Handle one JSON-RPC request; None for notifications."""
    method = request.get("method", "")
    req_id = request.get("id")
    if req_id is None:  # notification (e.g. notifications/initialized)
        return None
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION,
                  "capabilities": {"tools": {}},
                  "serverInfo": SERVER_INFO}
    elif method == "tools/list":
        result = {"tools": TOOL_DEFS}
    elif method == "tools/call":
        params = request.get("params", {})
        outcome = call_tool(params.get("name", ""),
                            params.get("arguments", {}), backend)
        result = {
            "content": [{"type": "text",
                         "text": json.dumps(outcome, ensure_ascii=False)}],
            "isError": not outcome.get("ok", False),
        }
    else:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def main() -> None:
    backend = make_backend()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(request, backend)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
