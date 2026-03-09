"""
Terminal tool — executes shell commands in a working directory.

This tool is the highest-risk tool in the framework.
It is registered with requires_confirm=True by default.

The router excludes any tool where requires_confirm=True from the
automatic tool spec sent to the model. Until a UI confirmation step
exists, terminal commands must be invoked explicitly by the caller
after the user has been shown the command and approved it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from auri.tools.base import BaseTool, ToolResult

_MAX_TIMEOUT = 120


class TerminalTool(BaseTool):
    name = "terminal"
    description = (
        "Run a shell command in the project directory. "
        "Use only for safe, non-destructive operations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default 30, max {_MAX_TIMEOUT}).",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    def __init__(self, working_dir: Path, requires_confirm: bool = True) -> None:
        self._cwd = working_dir
        # Checked by the router before including this tool in auto specs.
        # When True, the tool is never sent to the model automatically.
        self.requires_confirm = requires_confirm

    async def run(self, command: str, timeout: int = 30) -> ToolResult:
        timeout = min(max(1, timeout), _MAX_TIMEOUT)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout}s.",
            )
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(
            success=proc.returncode == 0,
            output={
                "command": command,
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            },
            error="" if proc.returncode == 0 else f"Exited with code {proc.returncode}",
        )
