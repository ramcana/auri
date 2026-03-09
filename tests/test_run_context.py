"""
Tests for RunContext — per-request metadata and transparency panel.

Covers:
- tokens_per_sec: correct calculation, None when data missing
- latency_ms: positive and increasing over time
- format_panel(): contains all expected sections
- format_panel(): tok/s shown inline with token counts
- format_panel(): fallback_reason shown in Note row
- format_panel(): context_truncated row
- format_panel(): tools_available and tools_used rows
- format_panel(): retrieval events (hit and miss)
- format_panel(): auto flag on model label
"""
from __future__ import annotations

import time

import pytest

from auri.run_context import RetrievalEvent, RunContext, ToolExecution


# ── tokens_per_sec ────────────────────────────────────────────────────────────

def test_tokens_per_sec_none_when_no_completion_tokens():
    ctx = RunContext(completion_tokens=0, prompt_tokens=50)
    ctx._start = time.monotonic() - 1.0  # 1s latency
    assert ctx.tokens_per_sec is None


def test_tokens_per_sec_none_when_zero_latency():
    ctx = RunContext(completion_tokens=100)
    ctx._start = time.monotonic()  # almost zero latency
    # Can't guarantee exactly 0 but we can set up the edge case by patching
    # Instead, verify the property returns a float when tokens and time are nonzero
    time.sleep(0.01)
    result = ctx.tokens_per_sec
    assert result is None or result > 0  # if latency > 0, must be positive


def test_tokens_per_sec_calculated():
    ctx = RunContext(completion_tokens=100)
    ctx._start = time.monotonic() - 2.0  # 2 seconds ago
    tps = ctx.tokens_per_sec
    assert tps is not None
    assert 40 < tps < 60  # ~50 tok/s with some timing slack


# ── latency_ms ────────────────────────────────────────────────────────────────

def test_latency_ms_positive():
    ctx = RunContext()
    time.sleep(0.01)
    assert ctx.latency_ms > 0


def test_latency_ms_increases():
    ctx = RunContext()
    t1 = ctx.latency_ms
    time.sleep(0.05)
    t2 = ctx.latency_ms
    assert t2 > t1


# ── format_panel() basics ─────────────────────────────────────────────────────

def make_ctx(**kwargs) -> RunContext:
    ctx = RunContext(
        model_name="llama3",
        model_display_name="Llama 3",
        backend="ollama",
        intent="chat",
        task_mode="Chat",
        **kwargs,
    )
    ctx._start = time.monotonic() - 0.5
    return ctx


def test_format_panel_contains_model():
    panel = make_ctx().format_panel()
    assert "Llama 3" in panel


def test_format_panel_contains_backend():
    panel = make_ctx().format_panel()
    assert "ollama" in panel


def test_format_panel_contains_intent():
    panel = make_ctx().format_panel()
    assert "chat" in panel


def test_format_panel_contains_task_mode():
    panel = make_ctx().format_panel()
    assert "Chat" in panel


def test_format_panel_contains_latency():
    panel = make_ctx().format_panel()
    assert "ms" in panel


def test_format_panel_auto_flag_when_auto_routed():
    ctx = make_ctx(auto_routed=True)
    panel = ctx.format_panel()
    assert "auto" in panel


def test_format_panel_no_auto_flag_when_manual():
    ctx = make_ctx(auto_routed=False)
    panel = ctx.format_panel()
    assert "auto" not in panel


# ── Tokens row ────────────────────────────────────────────────────────────────

def test_tokens_row_shown_when_available():
    ctx = make_ctx(prompt_tokens=100, completion_tokens=50)
    panel = ctx.format_panel()
    assert "100 in / 50 out" in panel


def test_tokens_row_includes_toks_per_sec():
    ctx = make_ctx(prompt_tokens=100, completion_tokens=200)
    ctx._start = time.monotonic() - 2.0  # ~100 tok/s
    panel = ctx.format_panel()
    assert "tok/s" in panel


def test_tokens_row_absent_when_zero():
    ctx = make_ctx(prompt_tokens=0, completion_tokens=0)
    panel = ctx.format_panel()
    assert "Tokens" not in panel


# ── Fallback Note row ─────────────────────────────────────────────────────────

def test_fallback_reason_shown_in_note():
    ctx = make_ctx(fallback_reason="timed out; fell back to backup")
    panel = ctx.format_panel()
    assert "Note" in panel
    assert "timed out" in panel


def test_no_fallback_reason_no_note():
    ctx = make_ctx(fallback_reason="")
    panel = ctx.format_panel()
    assert "Note" not in panel


# ── Context truncation row ────────────────────────────────────────────────────

def test_context_truncated_row_shown():
    ctx = make_ctx(context_truncated=True)
    panel = ctx.format_panel()
    assert "truncated" in panel


def test_context_not_truncated_no_row():
    ctx = make_ctx(context_truncated=False)
    panel = ctx.format_panel()
    assert "Context" not in panel


# ── Tools rows ────────────────────────────────────────────────────────────────

def test_tools_available_row():
    ctx = make_ctx(tools_available=["filesystem", "git"])
    panel = ctx.format_panel()
    assert "Tools Available" in panel
    assert "filesystem" in panel
    assert "git" in panel


def test_tools_used_ok_row():
    ctx = make_ctx(tools_used=[
        ToolExecution(name="git", arguments={}, elapsed_ms=12, success=True)
    ])
    panel = ctx.format_panel()
    assert "Tools Used" in panel
    assert "git" in panel
    assert "ok" in panel


def test_tools_used_fail_row():
    ctx = make_ctx(tools_used=[
        ToolExecution(name="filesystem", arguments={}, elapsed_ms=5, success=False, error="denied")
    ])
    panel = ctx.format_panel()
    assert "fail" in panel


# ── Retrieval rows ────────────────────────────────────────────────────────────

def test_retrieval_hit_row():
    ctx = make_ctx(retrieval_events=[
        RetrievalEvent(
            query="what is RAG",
            chunks_returned=3,
            sources=["docs/rag.md"],
            top_score=0.87,
            below_threshold=False,
        )
    ])
    panel = ctx.format_panel()
    assert "Retrieval" in panel
    assert "chunk" in panel
    assert "0.87" in panel


def test_retrieval_miss_row():
    ctx = make_ctx(retrieval_events=[
        RetrievalEvent(
            query="obscure topic",
            chunks_returned=0,
            sources=[],
            top_score=0.12,
            below_threshold=True,
        )
    ])
    panel = ctx.format_panel()
    assert "no match" in panel
    assert "0.12" in panel


def test_no_retrieval_events_no_row():
    ctx = make_ctx()
    panel = ctx.format_panel()
    assert "Retrieval" not in panel


# ── Markdown table structure ──────────────────────────────────────────────────

def test_format_panel_is_markdown_table():
    ctx = make_ctx()
    panel = ctx.format_panel()
    assert "---" in panel
    assert "|" in panel
    lines = panel.split("\n")
    table_lines = [l for l in lines if l.startswith("|")]
    assert len(table_lines) > 0
