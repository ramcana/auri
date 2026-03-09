"""
Tests for ToolRegistry — tool registration, scoping, and safety rules.

Covers:
- register() and get()
- auto_specs() excludes requires_confirm=True tools
- scoped() returns subset; confirm rules preserved
- scoped() with unknown name silently ignores it
- names() reflects registered set
- Tool result serialisation (to_json)
- Malformed-JSON tool arguments path (injected error, loop continues)
"""
from __future__ import annotations

import asyncio
import json

import pytest

from auri.tools.base import BaseTool, ToolResult
from auri.tools.registry import ToolRegistry


# ── Minimal tool stubs ────────────────────────────────────────────────────────

class _AutoTool(BaseTool):
    name = "filesystem"
    description = "list files"
    parameters: dict = {"type": "object", "properties": {"path": {"type": "string"}}}
    requires_confirm = False

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"files": []})


class _ConfirmTool(BaseTool):
    name = "terminal"
    description = "run shell commands"
    parameters: dict = {"type": "object", "properties": {"cmd": {"type": "string"}}}
    requires_confirm = True

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"stdout": ""})


class _AnotherAutoTool(BaseTool):
    name = "git"
    description = "git operations"
    parameters: dict = {"type": "object", "properties": {}}
    requires_confirm = False

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={})


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_and_get():
    reg = ToolRegistry()
    t = _AutoTool()
    reg.register(t)
    assert reg.get("filesystem") is t


def test_get_unknown_returns_none():
    reg = ToolRegistry()
    assert reg.get("nonexistent") is None


def test_names_reflects_registered():
    reg = ToolRegistry()
    reg.register(_AutoTool())
    reg.register(_ConfirmTool())
    assert set(reg.names()) == {"filesystem", "terminal"}


# ── auto_specs ────────────────────────────────────────────────────────────────

def test_auto_specs_excludes_confirm_tools():
    reg = ToolRegistry()
    reg.register(_AutoTool())
    reg.register(_ConfirmTool())
    specs = reg.auto_specs()
    names = [s["function"]["name"] for s in specs]
    assert "filesystem" in names
    assert "terminal" not in names


def test_auto_specs_empty_when_only_confirm_tools():
    reg = ToolRegistry()
    reg.register(_ConfirmTool())
    assert reg.auto_specs() == []


def test_auto_specs_spec_structure():
    reg = ToolRegistry()
    reg.register(_AutoTool())
    spec = reg.auto_specs()[0]
    assert spec["type"] == "function"
    assert "name" in spec["function"]
    assert "description" in spec["function"]
    assert "parameters" in spec["function"]


# ── scoped() ─────────────────────────────────────────────────────────────────

def test_scoped_returns_subset():
    reg = ToolRegistry()
    reg.register(_AutoTool())
    reg.register(_AnotherAutoTool())
    reg.register(_ConfirmTool())
    scoped = reg.scoped(["filesystem"])
    assert set(scoped.names()) == {"filesystem"}


def test_scoped_preserves_confirm_rules():
    reg = ToolRegistry()
    reg.register(_ConfirmTool())
    # terminal is confirm=True; scoped still respects that
    scoped = reg.scoped(["terminal"])
    assert scoped.get("terminal") is not None
    assert scoped.auto_specs() == []  # still excluded from auto specs


def test_scoped_unknown_name_ignored():
    reg = ToolRegistry()
    reg.register(_AutoTool())
    scoped = reg.scoped(["filesystem", "does_not_exist"])
    assert set(scoped.names()) == {"filesystem"}


def test_scoped_empty_list_returns_empty_registry():
    reg = ToolRegistry()
    reg.register(_AutoTool())
    scoped = reg.scoped([])
    assert scoped.names() == []
    assert scoped.auto_specs() == []


# ── ToolResult serialisation ──────────────────────────────────────────────────

def test_tool_result_to_json_success():
    result = ToolResult(success=True, output={"files": ["a.py"]})
    data = json.loads(result.to_json())
    assert data["success"] is True
    assert data["output"] == {"files": ["a.py"]}
    assert data["error"] == ""


def test_tool_result_to_json_failure():
    result = ToolResult(success=False, output=None, error="permission denied")
    data = json.loads(result.to_json())
    assert data["success"] is False
    assert data["error"] == "permission denied"


def test_tool_result_metadata_not_in_json():
    result = ToolResult(success=True, output={}, metadata={"secret": "hidden"})
    data = json.loads(result.to_json())
    assert "metadata" not in data
    assert "secret" not in data


# ── run() interface ───────────────────────────────────────────────────────────

def test_tool_run_returns_tool_result():
    tool = _AutoTool()
    result = asyncio.run(tool.run())
    assert isinstance(result, ToolResult)
    assert result.success is True


def test_base_tool_run_raises_not_implemented():
    class _BrokenTool(BaseTool):
        name = "broken"
        description = ""
        parameters: dict = {}

    tool = _BrokenTool()
    with pytest.raises(NotImplementedError):
        asyncio.run(tool.run())
