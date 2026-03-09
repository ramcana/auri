"""
TaskMode: declarative task mode definitions loaded from YAML.

Task modes live in configs/task_modes/*.yaml.
Each file defines what a mode enables:
  - display_name     — label shown in the UI sidebar
  - description      — one-line summary for documentation
  - preferred_caps   — capability tags that favour this mode in auto routing
  - enabled_tools    — tool names the model may call; empty = all auto tools
  - prompt_template  — template name from configs/prompts/
  - response_format  — "text" or "json"
  - max_output_tokens — output cap override; null = defer to template or sidebar

TaskModeLoader scans the directory, loads all YAML files, and supports
lookup by internal name (filename stem) or display name (UI label).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TaskMode:
    name: str                                      # filename stem (internal key)
    display_name: str                              # shown in the UI sidebar
    description: str = ""
    preferred_capabilities: list[str] = field(default_factory=list)
    enabled_tools: list[str] = field(default_factory=list)  # empty = all auto tools
    prompt_template: str = "chat"
    response_format: str = "text"                  # "text" | "json"
    max_output_tokens: Optional[int] = None        # None = defer to template/sidebar


class TaskModeLoader:
    """Loads TaskMode definitions from a directory of YAML files.

    Modes are ordered by their YAML filename (alphabetical) which determines
    the sidebar order. Name files accordingly (e.g., 01_chat.yaml) if explicit
    ordering matters.
    """

    _FALLBACK = TaskMode(
        name="chat",
        display_name="Chat",
        description="General conversation",
        prompt_template="chat",
    )

    def __init__(self, task_modes_dir: Path) -> None:
        self._dir = task_modes_dir
        self._modes: dict[str, TaskMode] = {}        # keyed by internal name
        self._by_display: dict[str, TaskMode] = {}   # keyed by display_name

    def load(self) -> None:
        """Scan task_modes_dir and load all *.yaml files."""
        if not self._dir.is_dir():
            logger.warning(
                "Task modes directory not found at %s — using built-in Chat fallback",
                self._dir,
            )
            self._modes = {self._FALLBACK.name: self._FALLBACK}
            self._by_display = {self._FALLBACK.display_name: self._FALLBACK}
            return

        modes: dict[str, TaskMode] = {}
        for path in sorted(self._dir.glob("*.yaml")):
            try:
                with path.open() as f:
                    data = yaml.safe_load(f) or {}
                name = path.stem
                raw_max = data.get("max_output_tokens")
                mode = TaskMode(
                    name=name,
                    display_name=data.get("display_name", name),
                    description=data.get("description", ""),
                    preferred_capabilities=list(data.get("preferred_capabilities") or []),
                    enabled_tools=list(data.get("enabled_tools") or []),
                    prompt_template=data.get("prompt_template", "chat"),
                    response_format=data.get("response_format", "text"),
                    max_output_tokens=int(raw_max) if raw_max is not None else None,
                )
                modes[name] = mode
                logger.debug("Loaded task mode: %s (%s)", name, mode.display_name)
            except Exception as exc:
                logger.warning("Failed to load task mode %s: %s", path, exc)

        if not modes:
            logger.warning("No task modes loaded — using built-in Chat fallback")
            modes[self._FALLBACK.name] = self._FALLBACK

        self._modes = modes
        self._by_display = {m.display_name: m for m in modes.values()}
        logger.info("TaskModeLoader loaded %d mode(s)", len(modes))

    def list_modes(self) -> list[TaskMode]:
        """Return all modes in YAML filename order (alphabetical)."""
        return list(self._modes.values())

    def display_names(self) -> list[str]:
        """Return display names in load order — used for sidebar widget values."""
        return [m.display_name for m in self._modes.values()]

    def get(self, name: str) -> Optional[TaskMode]:
        """Look up by internal name (YAML filename stem)."""
        return self._modes.get(name)

    def get_by_display(self, display_name: str) -> Optional[TaskMode]:
        """Look up by display name (value shown in the UI sidebar)."""
        return self._by_display.get(display_name)

    def resolve(self, key: str) -> Optional[TaskMode]:
        """Try internal name first, then display name — accepts either."""
        return self._modes.get(key) or self._by_display.get(key)
