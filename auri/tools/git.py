"""
Git tool — read-only git operations within a repository.

Allowed commands: status, diff, log, show.
Write operations (commit, push, reset, etc.) are not exposed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from auri.tools.base import BaseTool, ToolResult

_ALLOWED = frozenset({"status", "diff", "log", "show"})


class GitTool(BaseTool):
    name = "git"
    description = (
        "Run read-only git commands (status, diff, log, show) "
        "in the project repository."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["status", "diff", "log", "show"],
                "description": "Git subcommand to run.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional arguments (e.g. file paths, commit hashes, flags).",
                "default": [],
            },
        },
        "required": ["command"],
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    async def run(self, command: str, args: list[str] | None = None) -> ToolResult:
        if command not in _ALLOWED:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command '{command}' is not allowed. Allowed: {sorted(_ALLOWED)}",
            )

        cmd = ["git", command] + (args or [])
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            return ToolResult(success=False, output=None, error="Git command timed out.")
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        if proc.returncode != 0:
            return ToolResult(
                success=False,
                output=None,
                error=stderr.decode("utf-8", errors="replace").strip(),
            )

        return ToolResult(
            success=True,
            output={
                "command": " ".join(cmd),
                "output": stdout.decode("utf-8", errors="replace"),
            },
        )
