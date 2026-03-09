"""
Tests for failure-handling paths in ModelRouter.

These test the *internal* loop mechanics without a live vLLM/Ollama server.
The OpenAI client is replaced with an AsyncMock that returns controlled responses.

Covers:
  _tool_loop:
    - Malformed JSON in tool arguments → error injected, generation continues
    - Tool not found in registry → error injected, generation continues
    - Tool repetition guard → same (tool, args) skipped on second call
    - Tool exception → error injected, generation continues
    - Max tool iterations hit → falls through to final streaming pass
    - No tool calls in first response → content returned directly

  _stream_openai:
    - openai.APIError → yields error token
    - CancelledError → stops cleanly (no token yielded)

  _validate_lora:
    - Compatible LoRA → returned unchanged
    - Incompatible LoRA → returns None with warning logged

  auto_select:
    - exclude covers all candidates → returns None (tested in test_routing.py,
      repeated here as the timeout-fallback scenario)
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from auri.model_manager import ModelConfig, ModelManager, LoRAConfig
from auri.router import ModelRouter
from auri.run_context import RunContext
from auri.tools.base import BaseTool, ToolResult
from auri.tools.registry import ToolRegistry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _collect(agen) -> str:
    """Collect all tokens from an async generator into a single string."""
    async def _run():
        parts = []
        async for token in agen:
            parts.append(token)
        return "".join(parts)
    return asyncio.run(_run())


def make_router(models: list[ModelConfig] | None = None) -> ModelRouter:
    manager = MagicMock(spec=ModelManager)
    manager.list_models.return_value = models or []
    vllm = MagicMock()
    ollama = MagicMock()
    return ModelRouter(manager, vllm, ollama)


def make_tool_call(name: str, arguments: str, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def make_non_streaming_response(tool_calls=None, content=""):
    choice = MagicMock()
    choice.message.tool_calls = tool_calls
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    return response


def make_stream_chunk(content: str | None = None, usage=None):
    chunk = MagicMock()
    if content is not None:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
    else:
        chunk.choices = []
    chunk.usage = usage
    return chunk


class _OkTool(BaseTool):
    name = "filesystem"
    description = "list files"
    parameters: dict = {"type": "object", "properties": {"path": {"type": "string"}}}
    requires_confirm = False

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"files": ["a.py"]})


class _ErrorTool(BaseTool):
    name = "failing_tool"
    description = "always fails"
    parameters: dict = {"type": "object", "properties": {}}
    requires_confirm = False

    async def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("tool exploded")


def make_registry(*tools) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ── _tool_loop: malformed JSON arguments ─────────────────────────────────────

def test_malformed_json_args_injects_error_and_continues():
    router = make_router()

    # First pass: model returns a tool call with invalid JSON
    bad_tc = make_tool_call("filesystem", "NOT VALID JSON {{{")
    first_response = make_non_streaming_response(tool_calls=[bad_tc])

    # Second pass: model returns plain content (no more tool calls)
    second_response = make_non_streaming_response(tool_calls=None, content="Done.")

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[first_response, second_response]
    )

    registry = make_registry(_OkTool())
    ctx = RunContext(model_name="m")

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "list files"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=ctx,
    ))

    assert output == "Done."
    # RunContext should have recorded the failed tool call
    assert len(ctx.tools_used) == 1
    assert ctx.tools_used[0].success is False
    assert "malformed JSON" in ctx.tools_used[0].error


# ── _tool_loop: tool not in registry ─────────────────────────────────────────

def test_unknown_tool_injects_error():
    router = make_router()

    tc = make_tool_call("nonexistent_tool", '{"x": 1}')
    first_response = make_non_streaming_response(tool_calls=[tc])
    second_response = make_non_streaming_response(tool_calls=None, content="Recovered.")

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[first_response, second_response]
    )

    registry = make_registry(_OkTool())  # doesn't include nonexistent_tool
    ctx = RunContext(model_name="m")

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "do something"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=ctx,
    ))

    assert output == "Recovered."
    assert any(t.name == "nonexistent_tool" and not t.success for t in ctx.tools_used)


# ── _tool_loop: tool raises exception ────────────────────────────────────────

def test_tool_exception_injects_error_continues():
    router = make_router()

    tc = make_tool_call("failing_tool", '{}')
    first_response = make_non_streaming_response(tool_calls=[tc])
    second_response = make_non_streaming_response(tool_calls=None, content="Handled.")

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[first_response, second_response]
    )

    registry = make_registry(_ErrorTool())
    ctx = RunContext(model_name="m")

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "run failing tool"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=ctx,
    ))

    assert "Handled." in output
    assert any(t.name == "failing_tool" and not t.success for t in ctx.tools_used)


# ── _tool_loop: repetition guard ─────────────────────────────────────────────

def test_tool_repetition_guard_breaks_loop():
    router = make_router()

    # Model calls the same tool with the same args twice
    tc = make_tool_call("filesystem", '{"path": "/tmp"}', call_id="c1")
    tc2 = make_tool_call("filesystem", '{"path": "/tmp"}', call_id="c2")

    first_response = make_non_streaming_response(tool_calls=[tc])
    second_response = make_non_streaming_response(tool_calls=[tc2])

    # Final streaming response
    chunks = [make_stream_chunk("Final answer."), make_stream_chunk(None)]

    async def _stream(**kwargs):
        for c in chunks:
            yield c

    client = AsyncMock()
    # First two calls are non-streaming (tool loop); third is final streaming
    client.chat.completions.create = AsyncMock(
        side_effect=[first_response, second_response]
    )

    # Patch _stream_openai to return a simple async generator
    final_tokens = []

    async def fake_stream(client, model_api_name, messages, max_tokens, temperature, run_ctx=None):
        yield "Final answer."

    router._stream_openai = fake_stream

    registry = make_registry(_OkTool())

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "list files"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=None,
    ))

    # The repeated call should have been skipped; loop should have exited
    assert "Final answer." in output


# ── _tool_loop: no tool calls → direct content ───────────────────────────────

def test_no_tool_calls_returns_content_directly():
    router = make_router()

    response = make_non_streaming_response(tool_calls=None, content="Hello there.")
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    registry = make_registry(_OkTool())

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=None,
    ))

    assert output == "Hello there."


# ── _tool_loop: max iterations hit ───────────────────────────────────────────

def test_max_iterations_falls_through_to_stream():
    router = make_router()

    # Every non-streaming pass returns a tool call (never stops tool loop naturally)
    tc = make_tool_call("filesystem", '{"path": "/"}', call_id="cx")

    always_tool = make_non_streaming_response(tool_calls=[tc])
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=always_tool)

    async def fake_stream(client, model_api_name, messages, max_tokens, temperature, run_ctx=None):
        yield "final stream token"

    router._stream_openai = fake_stream

    registry = make_registry(_OkTool())

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "run"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=None,
    ))

    # After max_tool_iterations, streaming fallback should be called
    assert "final stream token" in output


# ── _tool_loop: non-streaming pass raises → falls back to stream ─────────────

def test_tool_loop_api_error_falls_back_to_stream():
    router = make_router()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=Exception("connection refused")
    )

    async def fake_stream(client, model_api_name, messages, max_tokens, temperature, run_ctx=None):
        yield "fallback stream"

    router._stream_openai = fake_stream

    registry = make_registry(_OkTool())

    output = _collect(router._tool_loop(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "run"}],
        max_tokens=256,
        temperature=0.0,
        tool_registry=registry,
        run_ctx=None,
    ))

    assert output == "fallback stream"


# ── _stream_openai: APIError ──────────────────────────────────────────────────

def test_stream_openai_api_error_yields_error_token():
    router = make_router()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.APIError("bad gateway", request=MagicMock(), body=None)
    )

    output = _collect(router._stream_openai(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.0,
    ))

    assert "Backend error" in output or "bad gateway" in output


# ── _stream_openai: CancelledError stops cleanly ─────────────────────────────

def test_stream_openai_cancelled_error_stops_cleanly():
    router = make_router()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    output = _collect(router._stream_openai(
        client=client,
        model_api_name="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.0,
    ))

    assert output == ""


# ── _validate_lora ────────────────────────────────────────────────────────────

def test_validate_lora_compatible_returns_name():
    router = make_router()
    model = ModelConfig(
        name="base", backend="vllm", display_name="Base",
        compatible_loras=["my-lora"],
    )
    result = router._validate_lora(model, "my-lora")
    assert result == "my-lora"


def test_validate_lora_incompatible_returns_none():
    router = make_router()
    model = ModelConfig(
        name="base", backend="vllm", display_name="Base",
        compatible_loras=["other-lora"],
    )
    # Manager returns None for lora lookup
    router._manager.get_lora.return_value = None

    result = router._validate_lora(model, "wrong-lora")
    assert result is None


def test_validate_lora_none_input_returns_none():
    router = make_router()
    model = ModelConfig(name="base", backend="vllm", display_name="Base")
    assert router._validate_lora(model, None) is None
