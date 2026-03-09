"""
ModelRouter: routes chat completion requests to vLLM or Ollama.

LoRA routing (vLLM):
    The vLLM OpenAI-compatible server distinguishes requests by the `model`
    field in the API call:
      - model = <served-model-name>  → base model, no LoRA
      - model = <lora-module-name>   → activates that LoRA adapter

    This behaviour should be verified with your specific vLLM version via the
    spike documented in the plan. The mapping is encapsulated here so it can be
    adjusted without changing any other file.

NOTE on LoRA naming: vLLM 0.9+ also lists loaded LoRA names as separate entries
in GET /v1/models. If your version uses a different naming convention, update
_resolve_vllm_model_name() accordingly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator, Optional

import openai

from auri.intent import Intent
from auri.model_manager import ModelConfig, ModelManager
from auri.ollama_client import OllamaClient
from auri.run_context import RunContext, RetrievalEvent, ToolExecution
from auri.vllm_server import VLLMServer, VLLMState

if TYPE_CHECKING:
    from auri.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(
        self,
        model_manager: ModelManager,
        vllm_server: VLLMServer,
        ollama_client: OllamaClient,
    ) -> None:
        self._manager = model_manager
        self._vllm = vllm_server
        self._ollama = ollama_client

    # ── Intent-based auto routing ─────────────────────────────────────────────

    # Rough chars-per-token heuristic for length filtering (no tokenizer needed)
    _CHARS_PER_TOKEN = 4
    # Messages longer than this estimated token count prefer long-context models
    _LONG_CONTEXT_THRESHOLD = 6000   # tokens
    _LONG_CONTEXT_MIN_LEN   = 32768  # model's max_model_len must be at least this

    def auto_select(
        self,
        intent: Intent,
        tool_registry: Optional["ToolRegistry"] = None,
        message_text: str = "",
        exclude: Optional[set] = None,
    ) -> Optional[ModelConfig]:
        """Return the best available model for the given intent.

        Filter stages (each stage narrows the candidate pool; if a stage
        would produce an empty pool it is skipped to guarantee a result):

        1. Attachment filter  — image → vision models; document → chat models
        2. Length filter      — long messages → models with large context windows
        3. Tool filter        — active tool specs → models with "tools" capability
        4. Intent filter      — preferred capability for the classified task
        5. Backend preference — Ollama first (always running), then vLLM

        Guaranteed: always returns a model as long as at least one is available.
        exclude: optional set of model names to skip (used for fallback selection).
        """
        candidates = [m for m in self._manager.list_models() if m.available]
        if exclude:
            candidates = [m for m in candidates if m.name not in exclude]
        if not candidates:
            return None

        def _narrow(pool: list, predicate) -> list:
            """Apply predicate; return original pool unchanged if result would be empty."""
            result = [m for m in pool if predicate(m)]
            return result if result else pool

        # Stage 1 — attachment type
        if intent.task == "vision":
            candidates = _narrow(candidates, lambda m: "vision" in m.capabilities)
        elif intent.task == "document":
            candidates = _narrow(candidates, lambda m: "chat" in m.capabilities)

        # Stage 2 — message length (long messages need large context windows)
        if message_text:
            estimated_tokens = len(message_text) // self._CHARS_PER_TOKEN
            if estimated_tokens > self._LONG_CONTEXT_THRESHOLD:
                candidates = _narrow(
                    candidates,
                    lambda m: m.max_model_len >= self._LONG_CONTEXT_MIN_LEN,
                )
                logger.debug(
                    "Long message (~%d tokens): preferring models with max_model_len >= %d",
                    estimated_tokens, self._LONG_CONTEXT_MIN_LEN,
                )

        # Stage 3 — tool requirement (only route to tool-capable models when tools are active)
        if tool_registry and tool_registry.auto_specs():
            candidates = _narrow(candidates, lambda m: "tools" in m.capabilities)

        # Stage 4 — intent capability preference
        cap_pref: dict[str, str] = {
            "coding":   "coding",
            "vision":   "vision",
            "document": "chat",
            "web":      "chat",
            "chat":     "chat",
        }
        preferred_cap = cap_pref.get(intent.task, "chat")
        candidates = _narrow(candidates, lambda m: preferred_cap in m.capabilities)

        # Stage 5 — prefer Ollama (always running) over vLLM
        ollama = [m for m in candidates if m.backend == "ollama"]
        return ollama[0] if ollama else candidates[0]

    # ── Main routing entry point ──────────────────────────────────────────────

    async def route_request(
        self,
        model_name: str,
        lora_name: Optional[str],
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tool_registry: Optional["ToolRegistry"] = None,
        run_ctx: Optional[RunContext] = None,
    ) -> AsyncIterator[str]:
        """Route a chat completion request and yield text tokens.

        When tool_registry is provided and the model has 'tools' capability,
        a tool execution loop runs before streaming the final response.
        run_ctx is populated with execution metadata as the request proceeds.
        """
        model_config = self._manager.get_model(model_name)
        if model_config is None:
            yield f"[Error: model '{model_name}' not found. Check models/ directory and models.yaml.]"
            return

        if not model_config.available:
            yield f"[Error: model '{model_config.display_name}' is offline. Start Ollama and restart Auri.]"
            return

        # Only offer tools to models that declare tool support
        active_registry = (
            tool_registry
            if tool_registry and "tools" in model_config.capabilities
            else None
        )

        if model_config.backend == "ollama":
            async for token in self._route_ollama(
                model_config, messages, max_tokens, temperature, active_registry, run_ctx
            ):
                yield token

        elif model_config.backend == "vllm":
            async for token in self._route_vllm(
                model_config, lora_name, messages, max_tokens, temperature, active_registry, run_ctx
            ):
                yield token

        else:
            yield f"[Error: unknown backend '{model_config.backend}' for model '{model_name}']"

    # ── Ollama routing ────────────────────────────────────────────────────────

    async def _route_ollama(
        self,
        model_config: ModelConfig,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tool_registry: Optional["ToolRegistry"] = None,
        run_ctx: Optional[RunContext] = None,
    ) -> AsyncIterator[str]:
        ollama_model = model_config.ollama_model_name or model_config.name
        logger.debug("Routing to Ollama: model='%s'", ollama_model)

        if tool_registry and tool_registry.auto_specs():
            async for token in self._tool_loop(
                client=self._ollama.client,
                model_api_name=ollama_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tool_registry=tool_registry,
                run_ctx=run_ctx,
            ):
                yield token
        else:
            async for token in self._stream_openai(
                client=self._ollama.client,
                model_api_name=ollama_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                run_ctx=run_ctx,
            ):
                yield token

    # ── vLLM routing ──────────────────────────────────────────────────────────

    async def _route_vllm(
        self,
        model_config: ModelConfig,
        lora_name: Optional[str],
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tool_registry: Optional["ToolRegistry"] = None,
        run_ctx: Optional[RunContext] = None,
    ) -> AsyncIterator[str]:
        # Validate LoRA compatibility
        validated_lora = self._validate_lora(model_config, lora_name)

        # Resolve which LoRAs to load: pinned + validated selection
        loras_to_load = self._manager.get_compatible_loras(model_config.name)
        # Filter to pinned + selected only (ensure_model handles LRU logic)
        pinned = {l.name for l in loras_to_load if l.name in model_config.pinned_loras}
        loras_for_server = [
            l for l in loras_to_load
            if l.name in pinned or l.name == validated_lora
        ]

        # Ensure vLLM is running with the right model/LoRAs.
        # If in FAILED state, ensure_model will attempt a restart via _restart_locked.
        try:
            await self._vllm.ensure_model(model_config, loras_for_server)
        except Exception as exc:
            logger.error("Failed to ensure vLLM model: %s", exc)
            yield f"[Error starting vLLM: {exc}]"
            return

        api_model_name = self._resolve_vllm_model_name(model_config, validated_lora)
        logger.debug("Routing to vLLM: model_api_name='%s'", api_model_name)

        vllm_client = self._vllm.get_openai_client()
        if tool_registry and tool_registry.auto_specs():
            async for token in self._tool_loop(
                client=vllm_client,
                model_api_name=api_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tool_registry=tool_registry,
                run_ctx=run_ctx,
            ):
                yield token
        else:
            async for token in self._stream_openai(
                client=vllm_client,
                model_api_name=api_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                run_ctx=run_ctx,
            ):
                yield token

    def _validate_lora(
        self,
        model_config: ModelConfig,
        lora_name: Optional[str],
    ) -> Optional[str]:
        """Return lora_name if compatible, else None (with a warning log).

        The caller is responsible for surfacing this warning to the user via
        a separate cl.Message before streaming begins.
        """
        if not lora_name:
            return None
        if lora_name in model_config.compatible_loras:
            return lora_name

        # Find the LoRA's declared base_model for a helpful error message
        lora_cfg = self._manager.get_lora(lora_name)
        required = lora_cfg.base_model if lora_cfg else "unknown"
        logger.warning(
            "LoRA '%s' (requires base='%s') is not compatible with model '%s'. "
            "Falling back to base model.",
            lora_name,
            required,
            model_config.name,
        )
        return None

    @staticmethod
    def _resolve_vllm_model_name(
        model_config: ModelConfig,
        lora_name: Optional[str],
    ) -> str:
        """Return the model string to pass in the OpenAI API `model` field.

        vLLM routes based on this field:
          - matches --served-model-name → base model
          - matches a --lora-modules name → activates that LoRA

        SPIKE NOTE: Verify this behaviour against your vLLM version.
        If LoRAs appear differently in /v1/models, adjust this mapping here.
        """
        if lora_name:
            return lora_name
        return model_config.name

    # ── Tool execution loop ───────────────────────────────────────────────────

    async def _tool_loop(
        self,
        client,
        model_api_name: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tool_registry: "ToolRegistry",
        run_ctx: Optional[RunContext] = None,
    ) -> AsyncIterator[str]:
        """Tool execution loop: non-streaming passes to detect and run tool calls,
        then a final streaming pass for the response.

        Safety limits:
          - max_tool_iterations = 3: prevents infinite tool call chains
          - all parse failures fall back to plain text — generation never crashes

        Populates run_ctx.tools_used with timing and success/failure per tool call.
        """
        tool_specs = tool_registry.auto_specs()
        aug = list(messages)
        max_tool_iterations = 3
        seen_calls: set[tuple[str, str]] = set()  # (fn_name, args_json) — repetition guard

        for _iteration in range(max_tool_iterations):
            try:
                response = await client.chat.completions.create(
                    model=model_api_name,
                    messages=aug,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tool_specs,
                    tool_choice="auto",
                    stream=False,
                )
            except Exception as exc:
                logger.warning(
                    "Tool call pass %d failed (%s) — falling back to plain streaming",
                    _iteration + 1, exc,
                )
                async for token in self._stream_openai(
                    client, model_api_name, aug, max_tokens, temperature, run_ctx=run_ctx
                ):
                    yield token
                return

            # Collect prompt tokens from the first non-streaming pass
            if run_ctx is not None and _iteration == 0 and response.usage:
                run_ctx.prompt_tokens = response.usage.prompt_tokens or 0

            choice = response.choices[0]

            if not choice.message.tool_calls:
                # No (more) tool calls — emit whatever content the model returned
                if choice.message.content:
                    # Capture completion tokens from this non-streaming pass
                    if run_ctx is not None and getattr(response, "usage", None):
                        run_ctx.completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                    yield choice.message.content
                    return
                # Model returned no content and no tool calls (e.g. Qwen3 think-only pass).
                # Fall through to the streaming pass so the model can produce a real answer.
                break

            # Append the assistant's tool-call message
            try:
                aug.append({
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls],
                })
            except Exception as exc:
                logger.warning("Failed to serialise tool_calls message (%s) — falling back", exc)
                if choice.message.content:
                    yield choice.message.content
                return

            # Execute each requested tool call
            stop_loop = False
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name

                try:
                    kwargs = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError as exc:
                    logger.warning("Tool '%s': malformed JSON arguments (%s)", fn_name, exc)
                    aug.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({
                            "error": (
                                f"Malformed JSON in tool arguments: {exc}. "
                                "Re-issue the call with valid JSON matching the schema."
                            )
                        }),
                    })
                    if run_ctx is not None:
                        run_ctx.tools_used.append(
                            ToolExecution(name=fn_name, arguments={}, elapsed_ms=0,
                                          success=False, error=f"malformed JSON: {exc}")
                        )
                    continue

                # Tool repetition guard — same tool+args already ran this request
                call_key = (fn_name, json.dumps(kwargs, sort_keys=True, default=str))
                if call_key in seen_calls:
                    logger.warning(
                        "Tool repetition detected: %s(%s) — breaking tool loop",
                        fn_name, call_key[1][:120],
                    )
                    stop_loop = True
                    aug.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "Tool call repeated — skipped to prevent loop."}),
                    })
                    continue
                seen_calls.add(call_key)

                tool = tool_registry.get(fn_name)

                if tool is None:
                    result_json = json.dumps({"error": f"Tool '{fn_name}' not available."})
                    if run_ctx is not None:
                        run_ctx.tools_used.append(
                            ToolExecution(name=fn_name, arguments={}, elapsed_ms=0,
                                          success=False, error="not available")
                        )
                else:
                    yield f"\n> `{fn_name}`\n"
                    t0 = time.monotonic()
                    try:
                        result = await tool.run(**kwargs)
                        result_json = result.to_json()
                        elapsed = int((time.monotonic() - t0) * 1000)
                        if run_ctx is not None:
                            run_ctx.tools_used.append(
                                ToolExecution(name=fn_name, arguments=kwargs,
                                              elapsed_ms=elapsed, success=result.success,
                                              error=result.error)
                            )
                            ev = result.metadata.get("retrieval_event")
                            if ev is not None:
                                run_ctx.retrieval_events.append(RetrievalEvent(**ev))
                    except Exception as exc:
                        elapsed = int((time.monotonic() - t0) * 1000)
                        result_json = json.dumps({"error": str(exc)})
                        if run_ctx is not None:
                            run_ctx.tools_used.append(
                                ToolExecution(name=fn_name, arguments=kwargs,
                                              elapsed_ms=elapsed, success=False, error=str(exc))
                            )

                aug.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })

            if stop_loop:
                break

        # Reached max iterations (or repetition guard triggered) — stream final response
        logger.debug("Tool loop ended after iteration %d — streaming final response", max_tool_iterations)
        async for token in self._stream_openai(
            client, model_api_name, aug, max_tokens, temperature, run_ctx=run_ctx
        ):
            yield token

    # ── Shared streaming helper ───────────────────────────────────────────────

    async def _stream_openai(
        self,
        client,
        model_api_name: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        run_ctx: Optional[RunContext] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completions from an OpenAI-compatible client, yielding text tokens.

        Requests usage stats via stream_options when run_ctx is provided.
        Backends that don't support stream_options silently ignore the field.
        """
        try:
            create_kwargs: dict = dict(
                model=model_api_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            if run_ctx is not None:
                create_kwargs["stream_options"] = {"include_usage": True}

            stream = await client.chat.completions.create(**create_kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                # Final chunk may carry usage (when stream_options was requested)
                if run_ctx is not None and getattr(chunk, "usage", None):
                    if chunk.usage.prompt_tokens:
                        run_ctx.prompt_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens:
                        run_ctx.completion_tokens = chunk.usage.completion_tokens

        except asyncio.CancelledError:
            # Client disconnected — stop cleanly
            logger.debug("Stream cancelled (client disconnected)")
            return

        except openai.APIError as exc:
            logger.error("Backend API error: %s", exc)
            yield f"\n\n[Backend error: {exc}]"

        except Exception as exc:
            logger.error("Unexpected error during streaming: %s", exc)
            yield f"\n\n[Unexpected error: {exc}]"

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_backend(self, model_name: str) -> Optional[str]:
        """Return 'vllm' or 'ollama' for the given model, or None if not found."""
        model = self._manager.get_model(model_name)
        return model.backend if model else None

    def incompatible_lora_message(
        self,
        model_name: str,
        lora_name: str,
    ) -> Optional[str]:
        """Return a human-readable warning string if lora_name is incompatible with model_name.

        Returns None if compatible (no warning needed).
        """
        model = self._manager.get_model(model_name)
        if model is None or lora_name in model.compatible_loras:
            return None
        lora = self._manager.get_lora(lora_name)
        required = lora.base_model if lora else "unknown"
        return (
            f"LoRA **{lora_name}** requires base model **{required}**, "
            f"but current model is **{model_name}**. "
            "Running without LoRA adapter."
        )
