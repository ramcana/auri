"""
Tests for ModelRouter.auto_select() — signal-driven model selection.

All tests run without a real vLLM server or Ollama daemon.
ModelManager is constructed with a pre-populated _models dict via monkeypatching.

Covers:
- Vision intent → vision-capable model preferred
- Coding intent → coding-capable model preferred
- Tool filter → tool-capable model required when specs are active
- Long-context filter → large-context model preferred
- Backend preference → Ollama before vLLM when both match
- Exclude parameter → skips named model (timeout fallback)
- Single available model always returned
- Returns None when no models available
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from auri.intent import Intent
from auri.model_manager import ModelConfig, ModelManager
from auri.router import ModelRouter
from auri.tools.registry import ToolRegistry
from auri.tools.base import BaseTool, ToolResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_model(
    name: str,
    backend: str = "ollama",
    capabilities: list[str] | None = None,
    max_model_len: int = 8192,
    available: bool = True,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        backend=backend,
        display_name=name,
        capabilities=capabilities or ["chat"],
        max_model_len=max_model_len,
        available=available,
    )


def make_router(models: list[ModelConfig]) -> ModelRouter:
    manager = MagicMock(spec=ModelManager)
    manager.list_models.return_value = models
    vllm = MagicMock()
    ollama = MagicMock()
    return ModelRouter(manager, vllm, ollama)


def chat_intent() -> Intent:
    return Intent(task="chat", signals=["default"])


def coding_intent() -> Intent:
    return Intent(task="coding", signals=["coding:debug"])


def vision_intent() -> Intent:
    return Intent(task="vision", signals=["image_attachment"])


# ── Basic selection ────────────────────────────────────────────────────────────

def test_returns_none_when_no_models():
    router = make_router([])
    assert router.auto_select(chat_intent()) is None


def test_returns_single_available_model():
    m = make_model("llama3", capabilities=["chat"])
    router = make_router([m])
    result = router.auto_select(chat_intent())
    assert result is m


def test_skips_unavailable_models():
    available = make_model("llama3", available=True)
    unavailable = make_model("codellama", available=False, capabilities=["coding"])
    router = make_router([available, unavailable])
    result = router.auto_select(coding_intent())
    # codellama is unavailable; falls back to llama3
    assert result is available


# ── Capability matching ────────────────────────────────────────────────────────

def test_vision_intent_prefers_vision_model():
    chat_m = make_model("llama3", capabilities=["chat"])
    vision_m = make_model("llava", capabilities=["chat", "vision"])
    router = make_router([chat_m, vision_m])
    result = router.auto_select(vision_intent())
    assert result is vision_m


def test_coding_intent_prefers_coding_model():
    chat_m = make_model("llama3", capabilities=["chat"])
    code_m = make_model("deepseek-coder", capabilities=["coding"])
    router = make_router([chat_m, code_m])
    result = router.auto_select(coding_intent())
    assert result is code_m


def test_falls_back_to_chat_when_no_coding_model():
    chat_m = make_model("llama3", capabilities=["chat"])
    router = make_router([chat_m])
    result = router.auto_select(coding_intent())
    # Stage 4 filter would leave empty pool → falls back to full pool → chat model
    assert result is chat_m


# ── Tool filter ────────────────────────────────────────────────────────────────

class _DummyTool(BaseTool):
    name = "dummy"
    description = "dummy"
    parameters: dict = {"type": "object", "properties": {}}
    requires_confirm = False

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={})


def make_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_DummyTool())
    return reg


def test_tool_filter_requires_tool_capable_model():
    chat_m = make_model("llama3", capabilities=["chat"])
    tools_m = make_model("mistral", capabilities=["chat", "tools"])
    router = make_router([chat_m, tools_m])
    result = router.auto_select(chat_intent(), tool_registry=make_tool_registry())
    assert result is tools_m


def test_tool_filter_falls_back_when_no_tool_model():
    chat_m = make_model("llama3", capabilities=["chat"])
    router = make_router([chat_m])
    # Stage 3 filter would empty pool → falls back → returns chat_m anyway
    result = router.auto_select(chat_intent(), tool_registry=make_tool_registry())
    assert result is chat_m


# ── Long-context filter ────────────────────────────────────────────────────────

def test_long_message_prefers_large_context_model():
    small_m = make_model("phi3", capabilities=["chat"], max_model_len=4096)
    large_m = make_model("llama3-70b", capabilities=["chat"], max_model_len=65536)
    router = make_router([small_m, large_m])
    # Threshold is 6000 tokens ~ 24000 chars
    long_text = "word " * 5000  # ~25000 chars
    result = router.auto_select(chat_intent(), message_text=long_text)
    assert result is large_m


def test_short_message_ignores_context_length():
    small_m = make_model("phi3", capabilities=["chat"], max_model_len=4096)
    large_m = make_model("llama3-70b", capabilities=["chat"], max_model_len=65536)
    router = make_router([small_m, large_m])
    result = router.auto_select(chat_intent(), message_text="Hello")
    # No long-context filter; both in pool; Ollama preference picks first
    assert result in (small_m, large_m)


# ── Backend preference ─────────────────────────────────────────────────────────

def test_ollama_preferred_over_vllm():
    vllm_m = make_model("deepseek", backend="vllm", capabilities=["chat"])
    ollama_m = make_model("llama3", backend="ollama", capabilities=["chat"])
    router = make_router([vllm_m, ollama_m])
    result = router.auto_select(chat_intent())
    assert result is ollama_m


# ── Exclude parameter (timeout fallback) ──────────────────────────────────────

def test_exclude_skips_named_model():
    m1 = make_model("llama3", backend="ollama", capabilities=["chat"])
    m2 = make_model("mistral", backend="ollama", capabilities=["chat"])
    router = make_router([m1, m2])
    result = router.auto_select(chat_intent(), exclude={"llama3"})
    assert result is m2


def test_exclude_all_returns_none():
    m1 = make_model("llama3")
    router = make_router([m1])
    result = router.auto_select(chat_intent(), exclude={"llama3"})
    assert result is None


def test_lora_selection_resolves_to_base_name():
    """_resolve_vllm_model_name: LoRA name takes precedence over base model name."""
    m = make_model("deepseek", backend="vllm")
    name = ModelRouter._resolve_vllm_model_name(m, "my-lora")
    assert name == "my-lora"


def test_no_lora_resolves_to_model_name():
    m = make_model("deepseek", backend="vllm")
    name = ModelRouter._resolve_vllm_model_name(m, None)
    assert name == "deepseek"
