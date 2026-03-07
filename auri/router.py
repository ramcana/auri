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
import logging
from typing import AsyncIterator, Optional

import openai

from auri.model_manager import ModelConfig, ModelManager
from auri.ollama_client import OllamaClient
from auri.vllm_server import VLLMServer, VLLMState

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

    async def route_request(
        self,
        model_name: str,
        lora_name: Optional[str],
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Route a chat completion request and yield text tokens.

        Handles both vLLM and Ollama backends transparently.
        """
        model_config = self._manager.get_model(model_name)
        if model_config is None:
            yield f"[Error: model '{model_name}' not found. Check models/ directory and models.yaml.]"
            return

        if not model_config.available:
            yield f"[Error: model '{model_config.display_name}' is offline. Start Ollama and restart Auri.]"
            return

        if model_config.backend == "ollama":
            async for token in self._route_ollama(model_config, messages, max_tokens, temperature):
                yield token

        elif model_config.backend == "vllm":
            async for token in self._route_vllm(model_config, lora_name, messages, max_tokens, temperature):
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
    ) -> AsyncIterator[str]:
        ollama_model = model_config.ollama_model_name or model_config.name
        logger.debug("Routing to Ollama: model='%s'", ollama_model)
        async for token in self._stream_openai(
            client=self._ollama.client,
            model_api_name=ollama_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
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

        # Ensure vLLM is running with the right model/LoRAs
        if self._vllm.state == VLLMState.FAILED:
            yield "[Error: vLLM server is in FAILED state. Check logs/active_vllm.json and logs/vllm_*.log]"
            return

        try:
            await self._vllm.ensure_model(model_config, loras_for_server)
        except Exception as exc:
            logger.error("Failed to ensure vLLM model: %s", exc)
            yield f"[Error starting vLLM: {exc}]"
            return

        api_model_name = self._resolve_vllm_model_name(model_config, validated_lora)
        logger.debug("Routing to vLLM: model_api_name='%s'", api_model_name)

        async for token in self._stream_openai(
            client=self._vllm.get_openai_client(),
            model_api_name=api_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
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

    # ── Shared streaming helper ───────────────────────────────────────────────

    async def _stream_openai(
        self,
        client,
        model_api_name: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream chat completions from an OpenAI-compatible client, yielding text tokens."""
        try:
            stream = await client.chat.completions.create(
                model=model_api_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

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
