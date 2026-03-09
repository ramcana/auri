"""
Tests for MetricsCollector and ModelMetrics.

Covers:
- record() increments request_count
- Success path: success_count, total_latency_ms, total_completion_tokens
- Failure path: timeout_count (not success_count)
- fallback_count incremented when fallback_reason contains "fell back to"
- tool_failure_count accumulated from run_ctx.tools_used
- retrieval_total and retrieval_below_threshold
- avg_latency_ms, avg_tokens_per_sec, retrieval_hit_rate, error_rate properties
- get() returns None for unknown model
- all() returns sorted snapshot
- Multiple models tracked independently
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from auri.metrics import MetricsCollector, ModelMetrics
from auri.run_context import RetrievalEvent, RunContext, ToolExecution


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_ctx(
    model_name: str = "llama3",
    latency_ms: int = 500,
    completion_tokens: int = 100,
    prompt_tokens: int = 50,
    fallback_reason: str = "",
    tools_used: list[ToolExecution] | None = None,
    retrieval_events: list[RetrievalEvent] | None = None,
) -> RunContext:
    ctx = RunContext(
        model_name=model_name,
        model_display_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        fallback_reason=fallback_reason,
        tools_used=tools_used or [],
        retrieval_events=retrieval_events or [],
    )
    # Patch latency_ms to return a fixed value
    ctx._start = time.monotonic() - latency_ms / 1000
    return ctx


def failed_tool() -> ToolExecution:
    return ToolExecution(name="filesystem", arguments={}, elapsed_ms=10, success=False, error="denied")


def ok_tool() -> ToolExecution:
    return ToolExecution(name="git", arguments={}, elapsed_ms=5, success=True)


def retrieval_hit() -> RetrievalEvent:
    return RetrievalEvent(query="q", chunks_returned=2, sources=["a.md"], top_score=0.8, below_threshold=False)


def retrieval_miss() -> RetrievalEvent:
    return RetrievalEvent(query="q", chunks_returned=0, sources=[], top_score=0.2, below_threshold=True)


# ── Basic record() ────────────────────────────────────────────────────────────

def test_first_record_creates_entry():
    col = MetricsCollector()
    ctx = make_ctx("llama3")
    col.record(ctx, success=True)
    m = col.get("llama3")
    assert m is not None
    assert m.request_count == 1


def test_success_increments_success_count():
    col = MetricsCollector()
    ctx = make_ctx("llama3", completion_tokens=100)
    col.record(ctx, success=True)
    m = col.get("llama3")
    assert m.success_count == 1
    assert m.total_completion_tokens == 100


def test_failure_increments_timeout_not_success():
    col = MetricsCollector()
    ctx = make_ctx("llama3")
    col.record(ctx, success=False)
    m = col.get("llama3")
    assert m.request_count == 1
    assert m.success_count == 0
    assert m.timeout_count == 1


def test_multiple_records_accumulate():
    col = MetricsCollector()
    for _ in range(3):
        col.record(make_ctx("llama3", latency_ms=200, completion_tokens=50), success=True)
    m = col.get("llama3")
    assert m.request_count == 3
    assert m.success_count == 3
    assert m.total_completion_tokens == 150


# ── Fallback counting ──────────────────────────────────────────────────────────

def test_fallback_count_incremented_on_fallback_reason():
    col = MetricsCollector()
    ctx = make_ctx("backup-model", fallback_reason="timed out; fell back to backup-model")
    col.record(ctx, success=True)
    m = col.get("backup-model")
    assert m.fallback_count == 1


def test_fallback_count_not_incremented_without_fell_back_to():
    col = MetricsCollector()
    ctx = make_ctx("llama3", fallback_reason="some other note")
    col.record(ctx, success=True)
    m = col.get("llama3")
    assert m.fallback_count == 0


def test_no_fallback_reason_no_fallback_count():
    col = MetricsCollector()
    ctx = make_ctx("llama3", fallback_reason="")
    col.record(ctx, success=True)
    assert col.get("llama3").fallback_count == 0


# ── Tool failure counting ──────────────────────────────────────────────────────

def test_tool_failure_count_accumulated():
    col = MetricsCollector()
    ctx = make_ctx("llama3", tools_used=[ok_tool(), failed_tool(), failed_tool()])
    col.record(ctx, success=True)
    assert col.get("llama3").tool_failure_count == 2


def test_no_tool_failures_zero():
    col = MetricsCollector()
    ctx = make_ctx("llama3", tools_used=[ok_tool()])
    col.record(ctx, success=True)
    assert col.get("llama3").tool_failure_count == 0


# ── Retrieval counting ────────────────────────────────────────────────────────

def test_retrieval_events_counted():
    col = MetricsCollector()
    ctx = make_ctx("llama3", retrieval_events=[retrieval_hit(), retrieval_miss()])
    col.record(ctx, success=True)
    m = col.get("llama3")
    assert m.retrieval_total == 2
    assert m.retrieval_below_threshold == 1


def test_retrieval_hit_rate_property():
    m = ModelMetrics("m", retrieval_total=4, retrieval_below_threshold=1)
    assert m.retrieval_hit_rate == pytest.approx(0.75)


def test_retrieval_hit_rate_none_when_no_retrievals():
    m = ModelMetrics("m")
    assert m.retrieval_hit_rate is None


# ── Computed properties ───────────────────────────────────────────────────────

def test_avg_latency_ms_none_when_no_successes():
    m = ModelMetrics("m")
    assert m.avg_latency_ms is None


def test_avg_latency_ms_computed():
    m = ModelMetrics("m", success_count=2, total_latency_ms=1000)
    assert m.avg_latency_ms == 500.0


def test_avg_tokens_per_sec_none_when_zero():
    m = ModelMetrics("m", success_count=1, total_latency_ms=0, total_completion_tokens=100)
    assert m.avg_tokens_per_sec is None


def test_avg_tokens_per_sec_computed():
    # 200 tokens over 2 seconds → 100 tok/s
    m = ModelMetrics("m", success_count=1, total_latency_ms=2000, total_completion_tokens=200)
    assert m.avg_tokens_per_sec == pytest.approx(100.0)


def test_error_rate_none_when_no_requests():
    m = ModelMetrics("m")
    assert m.error_rate is None


def test_error_rate_computed():
    m = ModelMetrics("m", request_count=4, timeout_count=1)
    assert m.error_rate == pytest.approx(0.25)


# ── get() and all() ───────────────────────────────────────────────────────────

def test_get_returns_none_for_unknown():
    col = MetricsCollector()
    assert col.get("nope") is None


def test_all_returns_sorted_by_name():
    col = MetricsCollector()
    for name in ["zmodel", "amodel", "mmodel"]:
        col.record(make_ctx(name), success=True)
    names = [m.model_name for m in col.all()]
    assert names == sorted(names)


def test_multiple_models_tracked_independently():
    col = MetricsCollector()
    col.record(make_ctx("model-a", completion_tokens=10), success=True)
    col.record(make_ctx("model-b", completion_tokens=20), success=False)
    a = col.get("model-a")
    b = col.get("model-b")
    assert a.success_count == 1
    assert a.timeout_count == 0
    assert b.success_count == 0
    assert b.timeout_count == 1


# ── log_summary() smoke test ──────────────────────────────────────────────────

def test_log_summary_no_crash(caplog):
    import logging
    col = MetricsCollector()
    col.record(make_ctx("llama3", latency_ms=300, completion_tokens=80), success=True)
    with caplog.at_level(logging.INFO):
        col.log_summary()  # should not raise


def test_log_summary_empty_no_crash(caplog):
    import logging
    col = MetricsCollector()
    with caplog.at_level(logging.INFO):
        col.log_summary()
