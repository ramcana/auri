"""
Prompt template loader.

Templates live in configs/prompts/*.yaml.
Each file defines: system_prompt, instructions, output_format.

Usage:
    library = PromptLibrary(settings.prompts_dir)
    library.load()
    template = library.get_or_default("summarize", default_system="You are Auri.")
    system_msg = template.build_system()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    name: str
    system_prompt: str
    instructions: str = ""
    output_format: str = ""
    response_format: str = "text"     # "text" | "json" — lets the router enforce structured output
    max_output_tokens: int | None = None  # per-template output cap; None = use model/sidebar default

    def build_system(self) -> str:
        """Assemble the full system prompt from all template sections."""
        parts = [self.system_prompt.strip()]
        if self.instructions.strip():
            parts.append(self.instructions.strip())
        if self.output_format.strip():
            parts.append(f"Output format:\n{self.output_format.strip()}")
        return "\n\n".join(parts)


class PromptLibrary:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._templates: dict[str, PromptTemplate] = {}

    def load(self) -> None:
        if not self._dir.is_dir():
            logger.debug("Prompts directory not found at %s — skipping", self._dir)
            return
        for path in sorted(self._dir.glob("*.yaml")):
            try:
                with path.open() as f:
                    data = yaml.safe_load(f) or {}
                raw_max = data.get("max_output_tokens")
                self._templates[path.stem] = PromptTemplate(
                    name=path.stem,
                    system_prompt=data.get("system_prompt", ""),
                    instructions=data.get("instructions", ""),
                    output_format=data.get("output_format", ""),
                    response_format=data.get("response_format", "text"),
                    max_output_tokens=int(raw_max) if raw_max is not None else None,
                )
                logger.debug("Loaded prompt template: %s", path.stem)
            except Exception as exc:
                logger.warning("Failed to load prompt template %s: %s", path, exc)

        logger.info("PromptLibrary loaded %d template(s)", len(self._templates))

    def get(self, name: str) -> Optional[PromptTemplate]:
        return self._templates.get(name)

    def get_or_default(self, name: str, default_system: str) -> PromptTemplate:
        t = self._templates.get(name)
        if t:
            return t
        return PromptTemplate(name=name, system_prompt=default_system)

    def list_names(self) -> list[str]:
        return list(self._templates.keys())
