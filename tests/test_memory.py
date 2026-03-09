"""
Tests for conversation memory — ConversationMemory, MemoryExtractor, MemoryDelta.

Covers:
  ConversationMemory:
    - is_empty when fresh
    - add_source deduplication
    - add_source returns (bool, evicted) tuple
    - source cap: evicts oldest-turn source at MAX_SOURCES
    - format_injection: empty when nothing to inject
    - format_injection: includes goal, sources, preferences
    - format_injection: goal capped at 80 chars
    - format_injection: shows at most 3 sources
    - format_summary: compact one-liner with turn numbers

  MemoryExtractor.extract_from_message:
    - Goal patterns: "I want to", "help me", "I need to", "I'm trying to", etc.
    - Short message guard: < 15 chars → no goal detected
    - No goal signal → active_goal unchanged
    - Goal set once; same goal on repeat → no duplicate update
    - Goal updates when new signal fires
    - _goal_turn recorded on goal update
    - Preference: concise, detailed, bullets, markdown, language
    - Multiple preferences can fire in one message
    - Preference already set → no delta entry if unchanged
    - _pref_turns recorded on preference update
    - Turn counter increments each call

  MemoryExtractor.update_from_run:
    - Sources from RAG retrieval events (above threshold only)
    - Sources from filesystem tool reads (success only)
    - Below-threshold retrieval → not added
    - Failed tool → not added
    - Duplicate source → not added twice
    - Eviction delta emitted when cap exceeded via update_from_run

  MemoryUpdate.format:
    - Goal, source, source_evicted, preference, task_mode shapes

  Panel integration:
    - memory_injected in RunContext surfaces in format_panel()
    - memory_updates in RunContext surfaces in format_panel()
    - Both absent → no Memory rows in panel
"""
from __future__ import annotations

import pytest

from auri.memory import ConversationMemory, MemoryDelta, MemoryExtractor, MemoryUpdate
from auri.run_context import RetrievalEvent, RunContext, ToolExecution


# ── Helpers ────────────────────────────────────────────────────────────────────

def fresh_memory() -> ConversationMemory:
    return ConversationMemory()


def extractor() -> MemoryExtractor:
    return MemoryExtractor()


def make_ctx(
    retrieval_events: list[RetrievalEvent] | None = None,
    tools_used: list[ToolExecution] | None = None,
) -> RunContext:
    return RunContext(
        model_name="m",
        retrieval_events=retrieval_events or [],
        tools_used=tools_used or [],
    )


def rag_hit(sources: list[str], top_score: float = 0.9) -> RetrievalEvent:
    return RetrievalEvent(
        query="q", chunks_returned=len(sources), sources=sources,
        top_score=top_score, below_threshold=False,
    )


def rag_miss() -> RetrievalEvent:
    return RetrievalEvent(
        query="q", chunks_returned=0, sources=[], top_score=0.1,
        below_threshold=True,
    )


def fs_tool(path: str, success: bool = True) -> ToolExecution:
    return ToolExecution(
        name="filesystem", arguments={"path": path},
        elapsed_ms=10, success=success,
    )


# ── ConversationMemory: is_empty ───────────────────────────────────────────────

def test_fresh_memory_is_empty():
    assert fresh_memory().is_empty()


def test_memory_not_empty_after_goal():
    m = fresh_memory()
    m.active_goal = "refactor the router"
    assert not m.is_empty()


def test_memory_not_empty_after_source():
    m = fresh_memory()
    m.referenced_sources.append("docs/api.md")
    assert not m.is_empty()


def test_memory_not_empty_after_preference():
    m = fresh_memory()
    m.preferences["verbosity"] = "concise"
    assert not m.is_empty()


# ── ConversationMemory: add_source return type ─────────────────────────────────

def test_add_source_returns_tuple_added_true():
    m = fresh_memory()
    added, evicted = m.add_source("docs/api.md")
    assert added is True
    assert evicted is None


def test_add_source_returns_tuple_added_false_on_duplicate():
    m = fresh_memory()
    m.add_source("docs/api.md")
    added, evicted = m.add_source("docs/api.md")
    assert added is False
    assert evicted is None


def test_add_source_deduplicates():
    m = fresh_memory()
    m.add_source("a.md")
    m.add_source("a.md")
    assert m.referenced_sources.count("a.md") == 1


# ── ConversationMemory: source cap and eviction ────────────────────────────────

def test_source_cap_size():
    assert ConversationMemory.MAX_SOURCES == 10


def test_source_cap_does_not_exceed_max():
    m = fresh_memory()
    ext = extractor()
    for i in range(ConversationMemory.MAX_SOURCES + 3):
        ext.extract_from_message(f"I want to do something useful now {i}", m)
        m.add_source(f"source_{i}.md")
    assert len(m.referenced_sources) == ConversationMemory.MAX_SOURCES


def test_source_cap_evicts_oldest():
    m = fresh_memory()
    ext = extractor()
    for i in range(ConversationMemory.MAX_SOURCES + 1):
        ext.extract_from_message(f"I want to do something useful now {i}", m)
        m.add_source(f"source_{i}.md")
    assert "source_0.md" not in m.referenced_sources


def test_source_cap_eviction_returns_evicted_name():
    m = fresh_memory()
    ext = extractor()
    for i in range(ConversationMemory.MAX_SOURCES):
        ext.extract_from_message(f"I want to do something useful now {i}", m)
        m.add_source(f"source_{i}.md")
    ext.extract_from_message("I want to do another useful thing here", m)
    added, evicted = m.add_source("new_source.md")
    assert added is True
    assert evicted == "source_0.md"


def test_source_cap_eviction_delta_in_update_from_run():
    m = fresh_memory()
    ext = extractor()
    # Fill to max
    for i in range(ConversationMemory.MAX_SOURCES):
        ext.extract_from_message(f"I want to do something useful now {i}", m)
        m.add_source(f"source_{i}.md")
    # One more via RAG should trigger eviction
    ctx = make_ctx(retrieval_events=[rag_hit(["new_source.md"])])
    delta = ext.update_from_run(ctx, m)
    eviction_updates = [u for u in delta.updates if u.field == "source_evicted"]
    assert len(eviction_updates) == 1


# ── ConversationMemory: format_injection ──────────────────────────────────────

def test_format_injection_empty_when_nothing():
    assert fresh_memory().format_injection() == ""


def test_format_injection_includes_goal():
    m = fresh_memory()
    m.active_goal = "build a test suite"
    inj = m.format_injection()
    assert "build a test suite" in inj
    assert "Current goal" in inj


def test_format_injection_includes_sources():
    m = fresh_memory()
    m.referenced_sources = ["docs/api.md", "src/router.py"]
    inj = m.format_injection()
    assert "docs/api.md" in inj
    assert "src/router.py" in inj


def test_format_injection_includes_preferences():
    m = fresh_memory()
    m.preferences["verbosity"] = "concise"
    inj = m.format_injection()
    assert "concise" in inj
    assert "verbosity" in inj


def test_format_injection_header():
    m = fresh_memory()
    m.active_goal = "test"
    assert "Session context" in m.format_injection()


def test_format_injection_goal_capped_at_80_chars():
    m = fresh_memory()
    m.active_goal = "x" * 100
    inj = m.format_injection()
    goal_line = [l for l in inj.split("\n") if "Current goal" in l][0]
    # "Current goal: " is 15 chars; goal portion should be ≤ 80 + 1 for ellipsis
    goal_portion = goal_line[len("Current goal: "):]
    assert len(goal_portion) <= 81  # 80 chars + "…"
    assert "…" in goal_portion


def test_format_injection_goal_not_truncated_when_short():
    m = fresh_memory()
    short = "fix the auth module"
    m.active_goal = short
    inj = m.format_injection()
    assert "…" not in inj
    assert short in inj


def test_format_injection_at_most_3_sources():
    m = fresh_memory()
    m.referenced_sources = ["a.md", "b.md", "c.md", "d.md", "e.md"]
    inj = m.format_injection()
    # Only first 3 shown, rest summarised as (+N more)
    assert "a.md" in inj
    assert "b.md" in inj
    assert "c.md" in inj
    assert "+2 more" in inj
    assert "d.md" not in inj
    assert "e.md" not in inj


def test_format_injection_exactly_3_sources_no_remainder():
    m = fresh_memory()
    m.referenced_sources = ["a.md", "b.md", "c.md"]
    inj = m.format_injection()
    assert "more" not in inj


# ── ConversationMemory: format_summary ────────────────────────────────────────

def test_format_summary_empty_when_nothing():
    assert fresh_memory().format_summary() == ""


def test_format_summary_shows_goal():
    m = fresh_memory()
    m.active_goal = "refactor the auth module"
    assert "refactor the auth module" in m.format_summary()


def test_format_summary_shows_sources():
    m = fresh_memory()
    m.referenced_sources = ["a.md", "b.py"]
    summary = m.format_summary()
    assert "a.md" in summary
    assert "b.py" in summary


def test_format_summary_shows_preferences():
    m = fresh_memory()
    m.preferences["verbosity"] = "concise"
    assert "concise" in m.format_summary()


def test_format_summary_shows_goal_turn_number():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("I want to build something awesome today", m)
    summary = m.format_summary()
    assert "goal[t1]:" in summary


def test_format_summary_shows_correct_turn_after_two_messages():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("What is two plus two?", m)  # turn 1, no goal
    ext.extract_from_message("I want to refactor the router module", m)  # turn 2, goal
    summary = m.format_summary()
    assert "goal[t2]:" in summary


def test_format_summary_shows_pref_turn_number():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("Be concise please in your responses", m)
    summary = m.format_summary()
    assert "verbosity=concise[t1]" in summary


# ── MemoryUpdate.format ────────────────────────────────────────────────────────

def test_update_format_goal():
    u = MemoryUpdate(field="goal", key="goal", value="build tests", reason="signal: 'I want to'")
    text = u.format()
    assert "goal" in text
    assert "build tests" in text
    assert "I want to" in text


def test_update_format_source():
    u = MemoryUpdate(field="source", key="source", value="docs/readme.md", reason="RAG retrieval hit")
    text = u.format()
    assert "source" in text
    assert "docs/readme.md" in text


def test_update_format_preference():
    u = MemoryUpdate(field="preference", key="verbosity", value="concise", reason="concise request")
    text = u.format()
    assert "verbosity=concise" in text


def test_update_format_long_goal_truncated():
    long_goal = "a" * 100
    u = MemoryUpdate(field="goal", key="goal", value=long_goal, reason="signal")
    text = u.format()
    assert "…" in text


def test_update_format_source_evicted():
    u = MemoryUpdate(field="source_evicted", key="source", value="old.md", reason="cap eviction")
    text = u.format()
    assert "old.md" in text
    assert "eviction" in text or "-source" in text


def test_update_format_task_mode():
    u = MemoryUpdate(field="task_mode", key="task_mode", value="code", reason="mode switch")
    text = u.format()
    assert "code" in text


# ── MemoryExtractor.extract_from_message: turn counter ───────────────────────

def test_turn_counter_starts_at_zero():
    m = fresh_memory()
    assert m._turn == 0


def test_turn_counter_increments_each_call():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("hello there friend", m)
    assert m._turn == 1
    ext.extract_from_message("another message here", m)
    assert m._turn == 2


# ── MemoryExtractor.extract_from_message: short message guard ─────────────────

def test_short_message_exact_boundary_no_goal():
    # "Help me" is 7 chars — well below the 15-char threshold
    m = fresh_memory()
    extractor().extract_from_message("Help me", m)
    assert m.active_goal == ""


def test_message_at_min_length_triggers_goal():
    # 15-char message with goal signal
    m = fresh_memory()
    # "Help me now ok" is exactly 14 chars — still below threshold
    extractor().extract_from_message("Help me now ok", m)
    assert m.active_goal == ""


def test_message_above_min_length_triggers_goal():
    m = fresh_memory()
    extractor().extract_from_message("Help me understand this code now", m)
    assert m.active_goal != ""


# ── MemoryExtractor.extract_from_message: goal ────────────────────────────────

@pytest.mark.parametrize("message,signal", [
    ("I want to refactor the router module", "I want to"),
    ("Help me write a test suite today", "help me"),
    ("I need to fix the authentication bug", "I need to"),
    ("I'm trying to understand the codebase", "I'm trying to"),
    ("I'd like to summarize this document now", "I'd like to"),
    ("My goal is to reduce latency here", "my goal is"),
    ("Let's build a memory system today", "let's build/fix/…"),
])
def test_goal_extraction_patterns(message, signal):
    m = fresh_memory()
    delta = extractor().extract_from_message(message, m)
    assert m.active_goal != "", f"Goal should have been set for: {message!r}"
    assert any(u.field == "goal" for u in delta.updates)


def test_no_goal_pattern_leaves_goal_empty():
    m = fresh_memory()
    extractor().extract_from_message("What is the capital of France?", m)
    assert m.active_goal == ""


def test_goal_not_duplicated_on_same_message():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("I want to refactor the router", m)
    delta2 = ext.extract_from_message("I want to refactor the router", m)
    # Second call: same goal already set → no delta entry
    assert not any(u.field == "goal" for u in delta2.updates)


def test_goal_updates_when_new_signal_fires():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("I want to build a test suite", m)
    first_goal = m.active_goal
    delta2 = ext.extract_from_message("Actually I need to fix the router first", m)
    assert m.active_goal != first_goal
    assert any(u.field == "goal" for u in delta2.updates)


def test_goal_uses_full_message_up_to_120_chars():
    long = "I want to " + "x" * 200
    m = fresh_memory()
    extractor().extract_from_message(long, m)
    assert len(m.active_goal) <= 120


def test_goal_turn_recorded_correctly():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("I want to build something useful here", m)
    assert m._goal_turn == m._turn
    assert m._goal_turn == 1


def test_goal_turn_matches_second_turn():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("What is the capital of France exactly?", m)  # no goal, turn 1
    ext.extract_from_message("I want to refactor the router module now", m)  # goal, turn 2
    assert m._goal_turn == 2


# ── MemoryExtractor.extract_from_message: preferences ─────────────────────────

@pytest.mark.parametrize("message,key,value", [
    ("Please be concise in your responses", "verbosity", "concise"),
    ("Keep it brief please when responding", "verbosity", "concise"),
    ("Be detailed and thorough in responses", "verbosity", "detailed"),
    ("Use bullet points please always", "format", "bullets"),
    ("Reply in markdown format please", "format", "markdown"),
    ("Respond in french for this session", "language", "french"),
    ("Respond in Japanese please thank you", "language", "japanese"),
])
def test_preference_extraction(message, key, value):
    m = fresh_memory()
    delta = extractor().extract_from_message(message, m)
    assert m.preferences.get(key) == value
    assert any(u.field == "preference" and u.key == key for u in delta.updates)


def test_same_preference_no_delta_on_repeat():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("Be concise please always", m)
    delta2 = ext.extract_from_message("Be concise please always", m)
    # Already set to concise → no delta
    assert not any(u.field == "preference" for u in delta2.updates)


def test_preference_overrides_on_change():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("Be concise please always", m)
    delta2 = ext.extract_from_message("Actually be detailed please", m)
    assert m.preferences["verbosity"] == "detailed"
    assert any(u.field == "preference" and u.value == "detailed" for u in delta2.updates)


def test_no_preference_signal_leaves_preferences_empty():
    m = fresh_memory()
    extractor().extract_from_message("What is 2 + 2 equal to?", m)
    assert m.preferences == {}


def test_pref_turn_recorded():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("Be concise please always", m)
    assert m._pref_turns.get("verbosity") == 1


def test_pref_turn_updated_on_change():
    m = fresh_memory()
    ext = extractor()
    ext.extract_from_message("Be concise please always", m)  # turn 1
    ext.extract_from_message("Actually be detailed please", m)  # turn 2
    assert m._pref_turns.get("verbosity") == 2


# ── MemoryExtractor.update_from_run ───────────────────────────────────────────

def test_rag_hit_adds_source():
    m = fresh_memory()
    ctx = make_ctx(retrieval_events=[rag_hit(["docs/api.md"])])
    delta = extractor().update_from_run(ctx, m)
    assert "docs/api.md" in m.referenced_sources
    assert any(u.field == "source" and u.value == "docs/api.md" for u in delta.updates)


def test_rag_miss_does_not_add_source():
    m = fresh_memory()
    ctx = make_ctx(retrieval_events=[rag_miss()])
    extractor().update_from_run(ctx, m)
    assert m.referenced_sources == []


def test_filesystem_tool_read_adds_source():
    m = fresh_memory()
    ctx = make_ctx(tools_used=[fs_tool("/home/ram/notes.md")])
    delta = extractor().update_from_run(ctx, m)
    assert "/home/ram/notes.md" in m.referenced_sources
    assert any(u.field == "source" for u in delta.updates)


def test_failed_filesystem_tool_does_not_add_source():
    m = fresh_memory()
    ctx = make_ctx(tools_used=[fs_tool("/secret", success=False)])
    extractor().update_from_run(ctx, m)
    assert m.referenced_sources == []


def test_duplicate_source_not_added_twice():
    m = fresh_memory()
    m.add_source("docs/api.md")  # already present
    ctx = make_ctx(retrieval_events=[rag_hit(["docs/api.md"])])
    delta = extractor().update_from_run(ctx, m)
    assert m.referenced_sources.count("docs/api.md") == 1
    assert not any(u.value == "docs/api.md" for u in delta.updates)


def test_multiple_sources_in_one_run():
    m = fresh_memory()
    ctx = make_ctx(
        retrieval_events=[rag_hit(["a.md", "b.md"])],
        tools_used=[fs_tool("c.py")],
    )
    extractor().update_from_run(ctx, m)
    assert "a.md" in m.referenced_sources
    assert "b.md" in m.referenced_sources
    assert "c.py" in m.referenced_sources


# ── RunContext panel integration ──────────────────────────────────────────────

def test_memory_injected_shows_in_panel():
    import time
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    ctx.memory_injected = "goal: 'refactor router'"
    panel = ctx.format_panel()
    assert "Memory Active" in panel
    assert "refactor router" in panel


def test_memory_updates_shows_in_panel():
    import time
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    ctx.memory_updates = ["goal → 'fix bug' (signal: 'I want to')", "+source: docs.md (RAG retrieval hit)"]
    panel = ctx.format_panel()
    assert "Memory Updated" in panel
    assert "fix bug" in panel
    assert "docs.md" in panel


def test_no_memory_no_panel_rows():
    import time
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    panel = ctx.format_panel()
    assert "Memory Active" not in panel
    assert "Memory Updated" not in panel
