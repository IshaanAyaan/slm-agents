"""MCP layer: tool schemas, guard enforcement, retry/fallback, JSON-RPC handling."""

from __future__ import annotations

import json

from slm_harness.mcp.server import handle
from slm_harness.mcp.tools import (
    GUARDS,
    TOOL_DEFS,
    RulesBackend,
    StubBackend,
    call_tool,
    run_specialist_task,
)
from slm_harness.subroutines.registry import REGISTRY


def test_every_subroutine_has_a_guard():
    assert set(GUARDS) == set(REGISTRY)


def test_tool_defs_have_json_schemas():
    names = {t["name"] for t in TOOL_DEFS}
    assert "slm_run_specialist_task" in names
    for t in TOOL_DEFS:
        assert t["inputSchema"]["type"] == "object"
        assert t["description"]


def test_repair_action_via_rules_backend():
    raw = "```json\n{\"action\": \"READ\", \"path\": \"src/a.py\", " \
          "\"offset\": 0, \"limit\": 40}\n```"
    res = call_tool("slm_repair_action", {"raw": raw}, RulesBackend())
    assert res["ok"]
    assert res["output"]["action"] == "READ"
    assert res["verified"] == "schema+guard"


def test_guard_rejects_unlisted_path_then_falls_back():
    # Stub returns a path that is not among the candidates: guard must reject it,
    # then the rules fallback must produce a guard-passing answer.
    stub = StubBackend('{"path": "evil/elsewhere.py"}')
    res = run_specialist_task(
        "path_normalizer",
        {"mention": "src\\app\\thing.py",
         "candidates": ["src/app/thing.py", "src/app/other.py"]},
        stub)
    assert res["ok"]
    assert res["backend"] == "rules-fallback"
    assert res["output"]["path"] == "src/app/thing.py"


def test_unknown_subroutine_and_tool():
    assert not run_specialist_task("nope", {}, StubBackend())["ok"]
    assert not call_tool("not_a_tool", {}, StubBackend())["ok"]


def test_jsonrpc_initialize_list_call():
    backend = RulesBackend()
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, backend)
    assert init["result"]["serverInfo"]["name"] == "slm-specialists"

    listed = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, backend)
    assert {t["name"] for t in listed["result"]["tools"]} == \
        {t["name"] for t in TOOL_DEFS}

    called = handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "slm_localize_failure", "arguments": {
            "traceback": ('Traceback (most recent call last):\n'
                          '  File "/proj/src/x.py", line 9, in <module>\n'
                          '    boom()\nKeyError: \'cfg\''),
            "project_prefix": "/proj/",
        }},
    }, backend)
    payload = json.loads(called["result"]["content"][0]["text"])
    assert payload["ok"]
    assert payload["output"] == {"path": "src/x.py", "line": 9,
                                 "category": "missing_key"}

    notif = handle({"jsonrpc": "2.0", "method": "notifications/initialized"},
                   backend)
    assert notif is None

    missing = handle({"jsonrpc": "2.0", "id": 4, "method": "bogus"}, backend)
    assert missing["error"]["code"] == -32601
