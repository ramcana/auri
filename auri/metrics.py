"""
MetricsCollector: in-memory runtime statistics per model.

Collected live after every request — never written to YAML or disk.
Designed to feed the router's routing decisions in a later sprint.

Tracked per model:
  request_count            — total inference requests routed to this model
  success_count            — completed without final timeout
  timeout_count            — all retry/fallback attempts exhausted (hard failure)
  fallback_count           — times this model was the *fallback* (not first choice)
  tool_failure_count       — sum of tool calls that returned an error across all requests
  total_latency_ms         — sum of successful request latencies (for avg)
  total_completion_tokens  — sum of completion tokens on successful requests (for tok/s)
  retrieval_total          — total retrieve_knowledge calls made via this model
  retrieval_below_threshold — calls where best score was below the similarity threshold
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from auri.run_context import RunContext

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    model_name: str
    request_count: int = 0
    success_count: int = 0
    timeout_count: int = 0          # final hard failures (all retries exhausted)
    fallback_count: int = 0         # times used as a fallback model
    tool_failure_count: int = 0
    total_latency_ms: int = 0
    total_completion_tokens: int = 0
    retrieval_total: int = 0
    retrieval_below_threshold: int = 0

    @property
    def avg_latency_ms(self) -> Optional[float]:
        """Average latency over successful requests. None if no successes yet."""
        if self.success_count == 0:
            return None
        return self.total_latency_ms / self.success_count

    @property
    def avg_tokens_per_sec(self) -> Optional[float]:
        """Average completion throughput over successful requests."""
        if self.total_latency_ms == 0 or self.total_completion_tokens == 0:
            return None
        return self.total_completion_tokens / (self.total_latency_ms / 1000)

    @property
    def retrieval_hit_rate(self) -> Optional[float]:
        """Fraction of retrieval calls that returned results above threshold."""
        if self.retrieval_total == 0:
            return None
        hits = self.retrieval_total - self.retrieval_below_threshold
        return hits / self.retrieval_total

    @property
    def error_rate(self) -> Optional[float]:
        """Fraction of requests that ended in hard failure (all retries exhausted)."""
        if self.request_count == 0:
            return None
        return self.timeout_count / self.request_count


class MetricsCollector:
    """Thread-safe in-memory store for per-model runtime metrics."""

    def __init__(self) -> None:
        self._metrics: dict[str, ModelMetrics] = {}
        self._lock = threading.Lock()

    def record(self, run_ctx: "RunContext", success: bool) -> None:
        """Record metrics for one completed inference request.

        success=True  — model responded before the timeout
        success=False — all retry/fallback attempts exhausted

        Metrics are attributed to run_ctx.model_name, which may be the
        fallback model's name when a fallback occurred.
        """
        name = run_ctx.model_name
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = ModelMetrics(model_name=name)
            m = self._metrics[name]

            m.request_count += 1

            if success:
                m.success_count += 1
                m.total_latency_ms += run_ctx.latency_ms
                m.total_completion_tokens += run_ctx.completion_tokens
            else:
                m.timeout_count += 1

            if run_ctx.fallback_reason and "fell back to" in run_ctx.fallback_reason:
                m.fallback_count += 1

            m.tool_failure_count += sum(
                1 for t in run_ctx.tools_used if not t.success
            )
            m.retrieval_total += len(run_ctx.retrieval_events)
            m.retrieval_below_threshold += sum(
                1 for r in run_ctx.retrieval_events if r.below_threshold
            )

        logger.debug(
            "Metrics recorded for '%s': success=%s latency=%dms tokens=%d",
            name, success, run_ctx.latency_ms, run_ctx.completion_tokens,
        )

    def get(self, model_name: str) -> Optional[ModelMetrics]:
        """Return current metrics for a model, or None if not yet seen."""
        with self._lock:
            return self._metrics.get(model_name)

    def all(self) -> list[ModelMetrics]:
        """Return a snapshot of all model metrics, sorted by model name."""
        with self._lock:
            return sorted(self._metrics.values(), key=lambda m: m.model_name)

    def log_summary(self) -> None:
        """Emit a human-readable summary of all model metrics at INFO level."""
        models = self.all()
        if not models:
            logger.info("Runtime metrics: no requests recorded yet")
            return
        logger.info("=== Runtime Metrics ===")
        for m in models:
            avg_lat = f"{m.avg_latency_ms:.0f}ms" if m.avg_latency_ms is not None else "n/a"
            tps     = f"{m.avg_tokens_per_sec:.1f} tok/s" if m.avg_tokens_per_sec is not None else "n/a"
            hit     = f"{m.retrieval_hit_rate:.0%}" if m.retrieval_hit_rate is not None else "n/a"
            logger.info(
                "  %-30s  %3d req  %3d ok  %2d timeout  avg %s  %s  retrieval hit %s",
                m.model_name, m.request_count, m.success_count,
                m.timeout_count, avg_lat, tps, hit,
            )
