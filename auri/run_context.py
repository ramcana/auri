"""
RunContext: collects per-request metadata for the transparency panel.

Created in app.py before each request, pre-populated with known fields
(intent, task mode, model), then passed to route_request() which fills in
tools_used, prompt_tokens, and completion_tokens as execution proceeds.

After streaming completes, call format_panel() to render the footer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolExecution:
    name: str
    arguments: dict
    elapsed_ms: int
    success: bool
    error: str = ""


@dataclass
class RetrievalEvent:
    query: str
    chunks_returned: int
    sources: list[str]       # distinct source paths returned
    top_score: float         # cosine similarity of best chunk
    below_threshold: bool = False  # True when all results were below min_score


@dataclass
class RunContext:
    intent: str = ""
    task_mode: str = ""
    model_name: str = ""
    model_display_name: str = ""
    backend: str = ""
    auto_routed: bool = False
    tools_available: list[str] = field(default_factory=list)
    tools_used: list[ToolExecution] = field(default_factory=list)
    retrieval_events: list[RetrievalEvent] = field(default_factory=list)
    # Token counts — populated by the router from API usage fields.
    # 0 means "not reported" (backend didn't return usage, or streaming without include_usage).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Set by app.py from ContextPacker.pack() result
    context_truncated: bool = False
    # Set when inference timed out and a retry or fallback model was used
    fallback_reason: str = ""
    # Workspace
    workspace_name: str = ""            # active workspace slug
    project_facts_count: int = 0        # facts injected from ProjectMemory this turn
    # Conversation memory: what was active going in, and what changed after
    memory_injected: str = ""           # non-empty summary if memory was injected
    memory_updates: list[str] = field(default_factory=list)  # formatted delta lines
    _start: float = field(default_factory=time.monotonic, repr=False, compare=False)

    @property
    def latency_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_sec(self) -> Optional[float]:
        """Completion throughput for this request. None when data is unavailable."""
        ms = self.latency_ms
        if self.completion_tokens == 0 or ms == 0:
            return None
        return self.completion_tokens / (ms / 1000)

    def format_panel(self) -> str:
        """Render the transparency panel as a compact markdown table."""
        rows: list[tuple[str, str]] = []

        if self.workspace_name:
            ws_label = f"`{self.workspace_name}`"
            if self.project_facts_count:
                ws_label += f" ({self.project_facts_count} project fact{'s' if self.project_facts_count != 1 else ''})"
            rows.append(("Workspace", ws_label))

        model_label = f"**{self.model_display_name}**" + (" *(auto)*" if self.auto_routed else "")
        rows.append(("Model", model_label))
        rows.append(("Backend", f"`{self.backend}`"))
        rows.append(("Intent", f"`{self.intent}`"))
        rows.append(("Task Mode", self.task_mode))
        rows.append(("Latency", f"{self.latency_ms} ms"))

        if self.prompt_tokens or self.completion_tokens:
            token_str = f"{self.prompt_tokens} in / {self.completion_tokens} out"
            if self.tokens_per_sec is not None:
                token_str += f" ({self.tokens_per_sec:.1f} tok/s)"
            rows.append(("Tokens", token_str))

        if self.context_truncated:
            rows.append(("Context", "truncated (history window applied)"))

        if self.fallback_reason:
            rows.append(("Note", f"⚠ {self.fallback_reason}"))

        if self.tools_available:
            rows.append(("Tools Available", ", ".join(f"`{t}`" for t in self.tools_available)))

        if self.tools_used:
            parts = []
            for t in self.tools_used:
                status = "ok" if t.success else "fail"
                parts.append(f"`{t.name}` [{status}, {t.elapsed_ms} ms]")
            rows.append(("Tools Used", ", ".join(parts)))

        if self.retrieval_events:
            parts = []
            for r in self.retrieval_events:
                q = r.query[:40] + "…" if len(r.query) > 40 else r.query
                if r.below_threshold:
                    parts.append(f'"{q}" → no match (best={r.top_score:.2f})')
                else:
                    srcs = ", ".join(f"`{s}`" for s in r.sources[:3])
                    parts.append(f'"{q}" → {r.chunks_returned} chunk(s) [{srcs}] (top={r.top_score:.2f})')
            rows.append(("Retrieval", " | ".join(parts)))

        if self.memory_injected:
            rows.append(("Memory Active", self.memory_injected))

        if self.memory_updates:
            rows.append(("Memory Updated", "; ".join(self.memory_updates)))

        lines = ["---", "| | |", "|---|---|"]
        for key, val in rows:
            lines.append(f"| {key} | {val} |")

        return "\n".join(lines)
