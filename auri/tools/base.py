"""
Base classes for the Auri tool framework.

Rule: tool outputs must always be JSON-serializable.
      ToolResult.to_json() enforces this at the boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any          # must be JSON-serializable
    error: str = ""
    # Internal metadata — NOT sent to the model. Tools can attach arbitrary
    # structured data here (e.g. retrieval events) for run_ctx recording.
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "success": self.success,
            "output": self.output,
            "error": self.error,
        })


class BaseTool:
    """Abstract base for all Auri tools."""

    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)  # JSON Schema for the tool's arguments

    async def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    def to_openai_spec(self) -> dict:
        """Return this tool as an OpenAI function-calling spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
