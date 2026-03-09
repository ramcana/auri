"""
Conversation memory — scoped, visible, per-session state.

Tracks four things and nothing else:
  active_goal         — what the user is trying to accomplish this session
  referenced_sources  — files/docs retrieved or read so far (capped at MAX_SOURCES)
  active_task_mode    — current task mode name
  preferences         — short-lived in-thread hints (verbosity, format, language)

Reset / staleness rules:
  goal:        new explicit goal signal always replaces the old one
  sources:     FIFO eviction when MAX_SOURCES is reached; oldest turn evicted first
  preferences: update only on an explicit signal in the current message
  all fields:  clear when the session ends (no cross-session persistence)

Turn tracking:
  _turn is incremented once per user message (via advance_turn()).
  _goal_turn, _source_turns, _pref_turns record which turn each field was last set.
  Turn numbers appear in format_summary() so the panel shows staleness at a glance.

Injection compactness rules:
  goal is capped at 80 chars in the injection string.
  Sources: at most 3 shown in injection regardless of how many are tracked.
  format_injection() must stay short — it is context, not instructions.

Design rules:
  - Memory is visible: every update carries a reason string
  - Memory is conservative: goal detection requires a minimum message length (15 chars)
  - Memory is auditable: format_summary() shows per-field turn numbers

MemoryExtractor.extract_from_message()  — called BEFORE inference
MemoryExtractor.update_from_run()       — called AFTER inference (sources only)
ConversationMemory.format_injection()   — appended to system prompt before packing
ConversationMemory.format_summary()     — compact string for the transparency panel
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from auri.run_context import RunContext

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class MemoryUpdate:
    """One change to memory, with the reason it was made."""
    field: str     # "goal" | "source" | "preference" | "task_mode"
    key: str       # field key (for preferences: the key; for others: same as field)
    value: str     # new value
    reason: str    # human-readable why

    def format(self) -> str:
        if self.field == "goal":
            truncated = self.value[:60] + "…" if len(self.value) > 60 else self.value
            return f"goal → '{truncated}' ({self.reason})"
        elif self.field == "source":
            return f"+source: {self.value} ({self.reason})"
        elif self.field == "source_evicted":
            return f"-source: {self.value} (cap eviction)"
        elif self.field == "preference":
            return f"{self.key}={self.value} ({self.reason})"
        elif self.field == "task_mode":
            return f"mode → {self.value}"
        return f"{self.field}: {self.value}"


@dataclass
class MemoryDelta:
    """Changes to memory from one extraction pass."""
    updates: list[MemoryUpdate] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.updates

    def format_lines(self) -> list[str]:
        return [u.format() for u in self.updates]


# ── ConversationMemory ─────────────────────────────────────────────────────────

@dataclass
class ConversationMemory:
    """Per-session memory. Created in on_chat_start; lives in cl.user_session."""

    # Hard cap on tracked sources. Oldest turn is evicted when exceeded.
    MAX_SOURCES: ClassVar[int] = 10

    # Injection display limits — keep the addendum compact.
    _INJECT_MAX_GOAL_CHARS: ClassVar[int] = 80
    _INJECT_MAX_SOURCES: ClassVar[int] = 3

    # Public state
    active_goal: str = ""
    referenced_sources: list[str] = field(default_factory=list)
    active_task_mode: str = ""
    preferences: dict[str, str] = field(default_factory=dict)

    # Turn tracking — internal, not injected
    _turn: int = field(default=0, repr=False, compare=False)
    _goal_turn: int = field(default=0, repr=False, compare=False)
    _source_turns: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _pref_turns: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def advance_turn(self) -> None:
        """Increment the turn counter. Call once per user message, before extraction."""
        self._turn += 1

    def is_empty(self) -> bool:
        return not (self.active_goal or self.referenced_sources or self.preferences)

    def add_source(self, source: str) -> tuple[bool, str | None]:
        """Add source. Evicts oldest if at MAX_SOURCES cap.

        Returns (added: bool, evicted: str | None).
        added=False when source was already present.
        evicted is set when an old source was removed to make room.
        """
        if source in self.referenced_sources:
            return False, None

        evicted: str | None = None
        if len(self.referenced_sources) >= self.MAX_SOURCES:
            # Evict the source added on the earliest turn (FIFO by turn)
            oldest_key = min(self._source_turns, key=lambda k: self._source_turns[k])
            self.referenced_sources.remove(oldest_key)
            del self._source_turns[oldest_key]
            evicted = oldest_key
            logger.debug("Memory: evicted source '%s' (cap=%d)", oldest_key, self.MAX_SOURCES)

        self.referenced_sources.append(source)
        self._source_turns[source] = self._turn
        return True, evicted

    def format_injection(self) -> str:
        """Return a compact system prompt addendum, or '' if nothing to inject.

        Hard limits: goal ≤ 80 chars, sources ≤ 3 shown.
        Kept short — this is context, not instructions.
        """
        parts = []
        if self.active_goal:
            goal_str = self.active_goal[:self._INJECT_MAX_GOAL_CHARS]
            if len(self.active_goal) > self._INJECT_MAX_GOAL_CHARS:
                goal_str += "…"
            parts.append(f"Current goal: {goal_str}")
        if self.referenced_sources:
            shown = self.referenced_sources[:self._INJECT_MAX_SOURCES]
            srcs = ", ".join(shown)
            remainder = len(self.referenced_sources) - len(shown)
            if remainder > 0:
                srcs += f" (+{remainder} more)"
            parts.append(f"Referenced files: {srcs}")
        if self.preferences:
            prefs = "; ".join(f"{k}: {v}" for k, v in self.preferences.items())
            parts.append(f"User preferences: {prefs}")
        if not parts:
            return ""
        return "--- Session context ---\n" + "\n".join(parts)

    def format_summary(self) -> str:
        """Compact one-liner for the transparency panel Memory row.

        Includes per-field turn numbers so staleness is visible at a glance.
        Example: "goal[t2]: 'refactor router' | sources[3]: a.md, b.md | verbosity=concise[t1]"
        """
        parts = []
        if self.active_goal:
            g = self.active_goal[:50] + "…" if len(self.active_goal) > 50 else self.active_goal
            parts.append(f"goal[t{self._goal_turn}]: '{g}'")
        if self.referenced_sources:
            srcs = ", ".join(self.referenced_sources[:3])
            count = len(self.referenced_sources)
            parts.append(f"sources[{count}]: {srcs}")
        if self.preferences:
            pref_parts = [
                f"{k}={v}[t{self._pref_turns.get(k, 0)}]"
                for k, v in self.preferences.items()
            ]
            parts.append("; ".join(pref_parts))
        return " | ".join(parts) if parts else ""


# ── MemoryExtractor ────────────────────────────────────────────────────────────

class MemoryExtractor:
    """Heuristic extractor — no LLM required.

    extract_from_message() — runs before inference; updates goal and preferences.
    update_from_run()      — runs after inference; updates sources from run_ctx.
    """

    # Minimum message length for goal detection.
    # Short messages ("ok", "yes", "continue") should not trigger goal updates.
    _GOAL_MIN_LEN: ClassVar[int] = 15

    # (regex, signal_name) — first match wins; no goal extracted if none fire
    _GOAL_PATTERNS: list[tuple[str, str]] = [
        (r"\bi want to\b",              "I want to"),
        (r"\bi(?:'m| am) trying to\b",  "I'm trying to"),
        (r"\bi need to\b",              "I need to"),
        (r"\bhelp me\b",               "help me"),
        (r"\bmy goal is\b",            "my goal is"),
        (r"\bi(?:'d| would) like to\b", "I'd like to"),
        (r"\bcan you\b.{0,30}\bfor me\b", "can you…for me"),
        (r"\blet(?:'s| us) (build|create|write|implement|fix|refactor)\b", "let's build/fix/…"),
    ]

    # (regex, pref_key, value_or_None, reason)
    # value=None means extract the matched word from the text
    _PREFERENCE_PATTERNS: list[tuple[str, str, str | None, str]] = [
        (r"\b(be concise|keep it (brief|short)|be brief|briefly)\b",
            "verbosity", "concise", "concise request"),
        (r"\b(be (detailed|thorough|comprehensive)|explain (fully|in detail|thoroughly))\b",
            "verbosity", "detailed", "detailed request"),
        (r"\buse bullet points?\b",
            "format", "bullets", "bullet point request"),
        (r"\bin markdown\b",
            "format", "markdown", "markdown request"),
        (r"\brespond in (english|french|spanish|german|italian|portuguese|japanese|chinese|korean)\b",
            "language", None, "language request"),
    ]

    def extract_from_message(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> MemoryDelta:
        """Extract goal and preferences from a user message. Mutates memory in-place.

        Called before inference. Returns what changed.
        Advances memory turn counter once per call.
        """
        memory.advance_turn()
        delta = MemoryDelta()
        lower = message.lower()

        # Goal detection — skip very short messages; first matching pattern wins
        if len(message) >= self._GOAL_MIN_LEN:
            for pattern, signal in self._GOAL_PATTERNS:
                if re.search(pattern, lower):
                    goal = message[:120].strip()
                    if goal != memory.active_goal:
                        memory.active_goal = goal
                        memory._goal_turn = memory._turn
                        delta.updates.append(MemoryUpdate(
                            field="goal",
                            key="goal",
                            value=goal,
                            reason=f"signal: '{signal}'",
                        ))
                        logger.debug("Memory: goal updated via '%s' at turn %d", signal, memory._turn)
                    break

        # Preference detection — multiple can fire per message
        for pattern, key, value, reason in self._PREFERENCE_PATTERNS:
            m = re.search(pattern, lower)
            if m:
                if value is None:
                    lang_m = re.search(
                        r"\b(english|french|spanish|german|italian|portuguese|japanese|chinese|korean)\b",
                        lower,
                    )
                    if lang_m:
                        value = lang_m.group(1)
                    else:
                        continue
                if memory.preferences.get(key) != value:
                    memory.preferences[key] = value
                    memory._pref_turns[key] = memory._turn
                    delta.updates.append(MemoryUpdate(
                        field="preference",
                        key=key,
                        value=value,
                        reason=reason,
                    ))
                    logger.debug("Memory: preference %s=%s at turn %d", key, value, memory._turn)

        return delta

    def update_from_run(
        self,
        run_ctx: "RunContext",
        memory: ConversationMemory,
    ) -> MemoryDelta:
        """Update memory from retrieval and tool events in a completed run.

        Called after inference. Returns what changed.
        Sources above threshold from RAG and successful filesystem reads are added.
        Evictions (when cap is hit) are included in the delta.
        """
        delta = MemoryDelta()

        def _add(source: str, reason: str) -> None:
            added, evicted = memory.add_source(source)
            if evicted:
                delta.updates.append(MemoryUpdate(
                    field="source_evicted",
                    key="source",
                    value=evicted,
                    reason="cap eviction",
                ))
            if added:
                delta.updates.append(MemoryUpdate(
                    field="source",
                    key="source",
                    value=source,
                    reason=reason,
                ))

        for event in run_ctx.retrieval_events:
            if not event.below_threshold:
                for source in event.sources:
                    _add(source, "RAG retrieval hit")

        for tool in run_ctx.tools_used:
            if tool.name == "filesystem" and tool.success:
                path = tool.arguments.get("path", "")
                if path:
                    _add(path, "filesystem tool read")

        return delta
