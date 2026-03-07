"""
Auri — Chainlit chat app with vLLM + Ollama dual inference backends.

Chainlit runs single-process by default, which is required to avoid GPU/port conflicts.
Start with: chainlit run app.py

Module-level singletons are shared across all user sessions.
Per-user state (message history, settings) lives in cl.user_session.
"""

from __future__ import annotations

import logging

import chainlit as cl
from chainlit.input_widget import Select, Slider

from auri.model_manager import ModelManager
from auri.ollama_client import OllamaClient
from auri.router import ModelRouter
from auri.settings import load_settings
from auri.vllm_server import VLLMServer

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("auri.app")

# ── Module-level singletons (created once, shared across sessions) ─────────────

_settings = load_settings()
logging.getLogger().setLevel(_settings.log_level)

_model_manager = ModelManager(_settings)
_model_manager.load()

_vllm_server = VLLMServer(_settings)
_ollama_client = OllamaClient(_settings)
_router = ModelRouter(_model_manager, _vllm_server, _ollama_client)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_model_display_values() -> list[str]:
    """Return model names for the Select widget (display_name not yet supported in Select)."""
    return _model_manager.list_model_names()


def _build_lora_values() -> list[str]:
    return ["None"] + _model_manager.list_lora_names()


def _model_label(name: str) -> str:
    model = _model_manager.get_model(name)
    if model is None:
        return name
    suffix = "" if model.available else " [offline]"
    return f"{model.display_name}{suffix}"


# ── Chainlit handlers ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    # 1. Probe Ollama availability and mark models accordingly
    ollama_ok = await _ollama_client.check_available()
    if ollama_ok:
        _model_manager.mark_ollama_available()
    else:
        _model_manager.mark_ollama_unavailable()
        logger.warning("Ollama is not running — Ollama-backend models are offline")

    # 2. Build lists for dropdowns
    model_names = _build_model_display_values()
    lora_names = _build_lora_values()

    if not model_names:
        await cl.Message(
            content=(
                "No models found. Add model directories to `models/vllm/` or `models/ollama/` "
                "and optionally configure them in `configs/models.yaml`."
            )
        ).send()
        return

    # 3. Send sidebar settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=model_names,
                initial_index=0,
            ),
            Select(
                id="lora",
                label="LoRA Adapter",
                values=lora_names,
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.7,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                initial=2048,
                min=128,
                max=8192,
                step=128,
            ),
        ]
    ).send()

    # 4. Initialise per-session state
    default_name = model_names[0]
    default_model = _model_manager.get_model(default_name)
    system_prompt = (
        default_model.system_prompt
        if default_model
        else "You are Auri, a helpful assistant."
    )

    cl.user_session.set("settings", settings)
    cl.user_session.set("message_history", [{"role": "system", "content": system_prompt}])

    # 5. Show offline warning if needed
    offline = [m.name for m in _model_manager.list_models() if not m.available]
    if offline:
        await cl.Message(
            content=f"Ollama is not running. The following models are offline: {', '.join(offline)}"
        ).send()

    logger.info("Chat session started — default model: %s", default_name)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    old_settings = cl.user_session.get("settings") or {}
    old_model_name = old_settings.get("model") if isinstance(old_settings, dict) else None
    new_model_name = settings.get("model")

    cl.user_session.set("settings", settings)

    # Update system prompt if model changed
    if new_model_name and new_model_name != old_model_name:
        new_model = _model_manager.get_model(new_model_name)
        if new_model:
            history = cl.user_session.get("message_history") or []
            if history and history[0].get("role") == "system":
                history[0]["content"] = new_model.system_prompt
            cl.user_session.set("message_history", history)

        # Pre-warm vLLM if the new model is vllm-backend
        if new_model and new_model.backend == "vllm":
            if not new_model.available:
                await cl.Message(content=f"Model **{new_model.display_name}** is not available.").send()
                return

            pinned = [
                _model_manager.get_lora(n)
                for n in new_model.pinned_loras
                if _model_manager.get_lora(n)
            ]

            status_msg = cl.Message(content=f"Loading **{new_model.display_name}**...")
            await status_msg.send()

            try:
                await _vllm_server.ensure_model(new_model, pinned)
                status_msg.content = f"**{new_model.display_name}** is ready."
            except Exception as exc:
                logger.error("Failed to load model '%s': %s", new_model_name, exc)
                status_msg.content = (
                    f"Failed to load **{new_model.display_name}**: {exc}\n\n"
                    f"Check `logs/vllm_*.log` for details."
                )
            await status_msg.update()

    logger.debug("Settings updated: model=%s lora=%s", new_model_name, settings.get("lora"))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = cl.user_session.get("settings") or {}
    history: list[dict] = cl.user_session.get("message_history") or []

    # Resolve current selections
    model_name: str = settings.get("model", "") if isinstance(settings, dict) else ""
    lora_name: str = settings.get("lora", "None") if isinstance(settings, dict) else "None"
    temperature: float = float(settings.get("temperature", 0.7) if isinstance(settings, dict) else 0.7)

    model_cfg = _model_manager.get_model(model_name)
    max_tokens: int = int(
        settings.get("max_tokens", model_cfg.max_tokens if model_cfg else 2048)
        if isinstance(settings, dict)
        else 2048
    )

    if not model_name or not model_cfg:
        await cl.Message(
            content="Please select a model in the settings panel (settings icon in the chat bar)."
        ).send()
        return

    # Warn about incompatible LoRA selection before streaming
    resolved_lora = lora_name if lora_name != "None" else None
    if resolved_lora:
        warning = _router.incompatible_lora_message(model_name, resolved_lora)
        if warning:
            await cl.Message(content=warning).send()
            resolved_lora = None  # router will also fall back, but be explicit here

    # Append user turn to history
    history.append({"role": "user", "content": message.content})

    # Stream response
    response_msg = cl.Message(content="")
    try:
        async for token in _router.route_request(
            model_name=model_name,
            lora_name=resolved_lora,
            messages=history,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            await response_msg.stream_token(token)
    finally:
        await response_msg.update()

    # Append assistant turn to history
    if response_msg.content:
        history.append({"role": "assistant", "content": response_msg.content})
    cl.user_session.set("message_history", history)


@cl.on_chat_end
async def on_chat_end() -> None:
    logger.debug("Chat session ended")
