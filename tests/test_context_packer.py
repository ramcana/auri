"""
Tests for ContextPacker — token-budget history packing.

Covers:
  - Empty history → system message only, not truncated
  - Single message → included; not truncated
  - Current turn always included (newest message first priority)
  - Oldest messages dropped when budget exceeded
  - truncated flag set correctly
  - history_turns_included and history_turns_total correct
  - System prompt always first in output messages
  - Tiny context_limit still includes system + current turn
  - output_budget reduces available space
  - Messages in chronological order in output
  - Multimodal content blocks handled without crash
  - estimate_tokens returns ≥ 1
"""
from __future__ import annotations

import pytest

from auri.context_packer import ContextPacker, PackedContext


SYSTEM = "You are Auri."


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def make_history(*pairs: tuple[str, str]) -> list[dict]:
    """pairs: (role, content) tuples in chronological order."""
    return [msg(role, content) for role, content in pairs]


# ── Empty history ─────────────────────────────────────────────────────────────

def test_empty_history_returns_system_only():
    packer = ContextPacker()
    result = packer.pack(SYSTEM, history=[], context_limit=4096)
    assert result.messages == [{"role": "system", "content": SYSTEM}]
    assert result.truncated is False
    assert result.history_turns_included == 0
    assert result.history_turns_total == 0


# ── Single message ────────────────────────────────────────────────────────────

def test_single_short_message_included():
    history = [msg("user", "Hello")]
    packer = ContextPacker()
    result = packer.pack(SYSTEM, history, context_limit=4096)
    assert len(result.messages) == 2
    assert result.messages[1]["content"] == "Hello"
    assert result.truncated is False
    assert result.history_turns_included == 1


# ── Current turn always included ──────────────────────────────────────────────

def test_current_turn_always_included_when_budget_tight():
    # Very small limit: enough for system + one short message
    long_old = "A" * 2000   # ~500 tokens
    short_new = "Hi"        # tiny
    history = [
        msg("user", long_old),
        msg("assistant", long_old),
        msg("user", short_new),  # current turn
    ]
    packer = ContextPacker()
    # context_limit=300 tokens, output_budget=100 → available=200 tokens
    result = packer.pack(SYSTEM, history, context_limit=300, output_budget=100)
    # Last message (short_new) must be present
    contents = [m["content"] for m in result.messages]
    assert short_new in contents


# ── Oldest messages dropped first ─────────────────────────────────────────────

def test_oldest_dropped_first():
    # Three turns; only the two most recent fit
    packer = ContextPacker()
    chunk = "word " * 100  # ~100 tokens each
    history = [
        msg("user", "old message " + chunk),
        msg("user", "middle " + chunk),
        msg("user", "newest"),
    ]
    # Limit tight enough to include newest + middle but not old
    # system ≈ 5 tokens, old ≈ 104, middle ≈ 104, newest ≈ 5
    # total needed for all: ~218; set limit to 150 tokens (output_budget=20 → avail=130)
    result = packer.pack(SYSTEM, history, context_limit=150, output_budget=20)
    contents = [m["content"] for m in result.messages]
    # "newest" must be present
    assert any("newest" in c for c in contents)
    # "old message" may be dropped
    if result.truncated:
        assert not any("old message" in c for c in contents)


def test_truncated_flag_set_when_history_dropped():
    packer = ContextPacker()
    big = "word " * 300  # ~300 tokens
    history = [msg("user", big), msg("user", big), msg("user", "current")]
    # Limit 200 tokens, budget 50 → available 150; two big messages won't fit
    result = packer.pack(SYSTEM, history, context_limit=200, output_budget=50)
    assert result.truncated is True
    assert result.history_turns_included < result.history_turns_total


def test_not_truncated_when_all_fit():
    packer = ContextPacker()
    history = [msg("user", "Hi"), msg("assistant", "Hello"), msg("user", "Bye")]
    result = packer.pack(SYSTEM, history, context_limit=8192)
    assert result.truncated is False
    assert result.history_turns_included == 3
    assert result.history_turns_total == 3


# ── history_turns counts ──────────────────────────────────────────────────────

def test_turns_total_always_reflects_input():
    packer = ContextPacker()
    history = [msg("user", "a"), msg("user", "b"), msg("user", "c")]
    result = packer.pack(SYSTEM, history, context_limit=8192)
    assert result.history_turns_total == 3


def test_turns_included_reflects_packed_count():
    packer = ContextPacker()
    big = "x " * 500
    history = [msg("user", big), msg("user", "short")]
    result = packer.pack(SYSTEM, history, context_limit=200, output_budget=50)
    # history_turns_included should equal len(result.messages) - 1 (system)
    assert result.history_turns_included == len(result.messages) - 1


# ── Message ordering ──────────────────────────────────────────────────────────

def test_system_prompt_is_first():
    history = [msg("user", "Hello"), msg("assistant", "Hi")]
    packer = ContextPacker()
    result = packer.pack(SYSTEM, history, context_limit=4096)
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == SYSTEM


def test_messages_in_chronological_order():
    packer = ContextPacker()
    history = [
        msg("user", "first"),
        msg("assistant", "second"),
        msg("user", "third"),
    ]
    result = packer.pack(SYSTEM, history, context_limit=4096)
    non_system = result.messages[1:]
    contents = [m["content"] for m in non_system]
    assert contents == ["first", "second", "third"]


# ── Token estimation ──────────────────────────────────────────────────────────

def test_estimate_tokens_minimum_one():
    packer = ContextPacker()
    assert packer.estimate_tokens("") >= 1
    assert packer.estimate_tokens("a") >= 1


def test_estimate_tokens_scales_with_length():
    packer = ContextPacker()
    short = packer.estimate_tokens("Hi")
    long = packer.estimate_tokens("word " * 1000)
    assert long > short


def test_estimated_tokens_in_result():
    packer = ContextPacker()
    result = packer.pack(SYSTEM, [msg("user", "Hello")], context_limit=4096)
    assert result.estimated_tokens > 0


# ── Output budget ─────────────────────────────────────────────────────────────

def test_large_output_budget_reduces_history_space():
    packer = ContextPacker()
    medium = "word " * 50  # ~50 tokens
    history = [msg("user", medium), msg("user", medium), msg("user", "current")]
    # Large output_budget eats into available space
    result_small_budget = packer.pack(SYSTEM, history, context_limit=512, output_budget=50)
    result_large_budget = packer.pack(SYSTEM, history, context_limit=512, output_budget=400)
    assert result_small_budget.history_turns_included >= result_large_budget.history_turns_included


# ── Multimodal content ────────────────────────────────────────────────────────

def test_multimodal_content_list_no_crash():
    packer = ContextPacker()
    multimodal_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ],
    }
    history = [multimodal_msg]
    result = packer.pack(SYSTEM, history, context_limit=4096)
    # Should not crash; the message should be included
    assert len(result.messages) == 2
    assert result.messages[1] is multimodal_msg


# ── Minimum available budget guard ────────────────────────────────────────────

def test_tiny_context_limit_does_not_crash():
    packer = ContextPacker()
    history = [msg("user", "hello")]
    # context_limit - output_budget < 256 → should clamp to 256
    result = packer.pack(SYSTEM, history, context_limit=100, output_budget=200)
    # Should not crash; system message always present
    assert result.messages[0]["role"] == "system"
