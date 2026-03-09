"""
Filesystem tool — sandboxed read/list within the workspace root.

Only "read" and "list" are exposed. Write operations are intentionally
excluded until a project workspace + permission model exists.

Path traversal is prevented: any path that resolves outside the sandbox
root is rejected before any I/O occurs.
"""

from __future__ import annotations

from pathlib import Path

from auri.tools.base import BaseTool, ToolResult


class FilesystemTool(BaseTool):
    name = "filesystem"
    description = (
        "Read a file or list directory contents within the user's project workspace. "
        "Use this to access project source code, configs, or notes for tasks like "
        "code review, debugging, or explaining code. "
        "Paths must be relative (e.g. 'auri/model_manager.py', 'README.md', '.'). "
        "Do NOT use this to explore system directories or look for runtime information."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list"],
                "description": "Action to perform: 'read' a file or 'list' a directory.",
            },
            "path": {
                "type": "string",
                "description": "Relative path within the project workspace (e.g. 'src/app.py' or '.').",
            },
        },
        "required": ["action", "path"],
    }

    def __init__(self, sandbox_root: Path) -> None:
        self._root = sandbox_root.resolve()

    def _safe_resolve(self, path_str: str) -> Path | None:
        """Resolve path and confirm it stays within the sandbox."""
        target = (self._root / path_str).resolve()
        if not str(target).startswith(str(self._root)):
            return None
        return target

    async def run(self, action: str, path: str) -> ToolResult:
        target = self._safe_resolve(path)
        if target is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Path '{path}' is outside the allowed workspace.",
            )

        if action == "read":
            if not target.exists():
                return ToolResult(success=False, output=None, error=f"File not found: {path}")
            if not target.is_file():
                return ToolResult(success=False, output=None, error=f"Not a file: {path}")
            try:
                content = target.read_text(encoding="utf-8", errors="replace")
                return ToolResult(success=True, output={"path": path, "content": content})
            except Exception as exc:
                return ToolResult(success=False, output=None, error=str(exc))

        if action == "list":
            if not target.exists():
                return ToolResult(success=False, output=None, error=f"Not found: {path}")
            if not target.is_dir():
                return ToolResult(success=False, output=None, error=f"Not a directory: {path}")
            entries = [
                {
                    "name": e.name,
                    "type": "dir" if e.is_dir() else "file",
                    "size": e.stat().st_size if e.is_file() else None,
                }
                for e in sorted(target.iterdir())
            ]
            return ToolResult(success=True, output={"path": path, "entries": entries})

        return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
