"""Regression: a model-emitted invalid file glob must not crash the eval.

During the 2026-06-10 real run, a fine-tuned student emitted a glob like
"**.py"; pathlib raised ValueError inside the python grep fallback, escaped the
harness error handling, and killed the whole run_conditions process mid-C4.
The fix surfaces it as a normal is_error ToolResult so the harness retry loop
can feed it back to the model.
"""

from __future__ import annotations

import pytest

from openharness.tools import grep_tool as grep_mod
from openharness.tools.base import ToolExecutionContext
from openharness.tools.grep_tool import GrepTool, GrepToolInput


@pytest.mark.asyncio
async def test_invalid_glob_returns_tool_error_not_crash(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("class GatewayState: ...\n", encoding="utf-8")

    # Force the python fallback (the path that crashed), regardless of whether
    # ripgrep is installed on the test machine.
    async def no_rg(**kwargs):
        return None

    monkeypatch.setattr(grep_mod, "_rg_grep", no_rg)

    # Python 3.12 pathlib raises ValueError for "**.py"; 3.13+ accepts it.
    # Either way the tool must complete and never propagate the exception.
    try:
        list(tmp_path.glob("**.py"))
        glob_raises = False
    except ValueError:
        glob_raises = True

    tool = GrepTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        GrepToolInput(pattern="GatewayState", file_glob="**.py"), ctx
    )
    if glob_raises:
        assert result.is_error
        assert "invalid file glob" in result.output
    else:
        assert not result.is_error

    # A valid glob through the same fallback still works.
    ok = await tool.execute(
        GrepToolInput(pattern="GatewayState", file_glob="**/*.py"), ctx
    )
    assert not ok.is_error
    assert "module.py" in ok.output


@pytest.mark.asyncio
async def test_glob_tool_invalid_pattern_returns_tool_error(tmp_path):
    """Same crash class via the standalone glob tool (hit during the H100 run:
    glob_tool.py's python fallback let pathlib's ValueError kill the C4 eval)."""
    from openharness.tools.glob_tool import GlobTool, GlobToolInput

    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    try:
        list(tmp_path.glob("**.py"))
        glob_raises = False
    except ValueError:
        glob_raises = True

    tool = GlobTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(GlobToolInput(pattern="**.py"), ctx)
    if glob_raises:
        assert result.is_error
        assert "invalid glob pattern" in result.output
    else:
        assert not result.is_error

    ok = await tool.execute(GlobToolInput(pattern="**/*.py"), ctx)
    assert not ok.is_error
    assert "module.py" in ok.output
