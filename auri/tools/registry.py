"""
ToolRegistry — holds all registered tools and produces OpenAI tool specs.

Tools with requires_confirm=True (e.g. TerminalTool) are registered
but excluded from auto_specs() so they are never sent to the model
without explicit user approval.
"""

from __future__ import annotations

from typing import Optional

from auri.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def auto_specs(self) -> list[dict]:
        """OpenAI tool specs for tools that do NOT require user confirmation."""
        return [
            t.to_openai_spec()
            for t in self._tools.values()
            if not getattr(t, "requires_confirm", False)
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def scoped(self, allowed_names: list[str]) -> "ToolRegistry":
        """Return a new registry containing only the named tools (auto-confirm rules preserved).

        Used by task modes to restrict which tools the model may call.
        Tools not in allowed_names are simply absent from the returned registry.
        """
        allowed = set(allowed_names)
        scoped_registry = ToolRegistry()
        for name, tool in self._tools.items():
            if name in allowed:
                scoped_registry.register(tool)
        return scoped_registry
