"""
Auri — Chainlit chat app with vLLM + Ollama dual inference backends.

Chainlit runs single-process by default, which is required to avoid GPU/port conflicts.
Start with: chainlit run app.py

Module-level singletons are shared across all user sessions.
Per-user state (message history, settings, active workspace, tool registry) lives in cl.user_session.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider

from auri.context_packer import ContextPacker
from auri.intent import IntentClassifier
from auri.memory import ConversationMemory, MemoryExtractor
from auri.metrics import MetricsCollector
from auri.model_manager import ModelManager, ValidationIssue
from auri.ollama_client import OllamaClient
from auri.prompts import PromptLibrary
from auri.router import ModelRouter
from auri.run_context import RunContext
from auri.task_mode import TaskModeLoader
from auri.settings import load_settings
from auri.rag.embedder import Embedder
from auri.rag.ingest import Ingestor, classify_file_type
from auri.rag.retriever import Retriever
from auri.rag.store import VectorStore
from auri.tools.filesystem import FilesystemTool
from auri.tools.git import GitTool
from auri.tools.registry import ToolRegistry
from auri.tools.retrieval import RetrievalTool
from auri.tools.terminal import TerminalTool
from auri.tools.web import WebSearchTool
from auri.vllm_server import VLLMServer
from auri.workspace import ProjectMemory, Workspace, WorkspaceManager

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

_intent_classifier = IntentClassifier()

_prompt_library = PromptLibrary(_settings.prompts_dir)
_prompt_library.load()

_task_mode_loader = TaskModeLoader(_settings.project_root / "configs" / "task_modes")
_task_mode_loader.load()

_context_packer = ContextPacker()

_metrics_collector = MetricsCollector()
_memory_extractor = MemoryExtractor()

# Embedder is stateless and expensive to initialise — shared across workspaces.
_embedder = Embedder()

# Workspace manager: creates/lists workspace directories under workspaces/
_workspace_manager = WorkspaceManager(_settings.workspaces_root)
_workspace_manager.ensure_default()

# ── Constants ─────────────────────────────────────────────────────────────────

_AUTO_LABEL = "Auto (intent-based)"

# Per-attempt inference timeout. On timeout: retry once with same model,
# then try the next-best model, then surface a hard error.
_INFERENCE_TIMEOUT_S = 120

# Intent task → task mode display name used when Auto routing is active.
# (does not change the sidebar selection; only affects this turn)
_INTENT_TO_MODE = {
    "coding":   "Code Review",
    "vision":   "Chat",
    "document": "Summarize",
    "web":      "Chat",
    "chat":     "Chat",
}


# ── Workspace session helpers ─────────────────────────────────────────────────

def _build_workspace_session(workspace: Workspace) -> tuple[ToolRegistry, Ingestor]:
    """Build a fresh ToolRegistry and Ingestor scoped to the given workspace.

    Each workspace gets its own VectorStore (isolated knowledge base) and its
    own filesystem sandbox (workspace files/ directory).
    Called on chat start and every time the user switches workspace.
    """
    ws_store = VectorStore(workspace.knowledge_dir)
    ws_retriever = Retriever(_embedder, ws_store)
    ws_ingestor = Ingestor(_embedder, ws_store)

    registry = ToolRegistry()
    registry.register(FilesystemTool(sandbox_root=workspace.files_dir))
    registry.register(WebSearchTool())
    registry.register(GitTool(repo_root=_settings.project_root))
    registry.register(TerminalTool(working_dir=_settings.project_root, requires_confirm=True))
    registry.register(RetrievalTool(ws_retriever))

    return registry, ws_ingestor


# ── Other helpers ─────────────────────────────────────────────────────────────

def _model_selector_values() -> list[str]:
    """Model names for the Select widget, with Auto as the first entry."""
    return [_AUTO_LABEL] + _model_manager.list_model_names()


def _lora_values() -> list[str]:
    return ["None"] + _model_manager.list_lora_names()


def _system_prompt_for(model_name: str | None, task_mode: str) -> str:
    """Build the system prompt from the prompt template for the given task mode."""
    mode = _task_mode_loader.resolve(task_mode)
    template_name = mode.prompt_template if mode else "chat"
    default_system = "You are Auri, a helpful assistant."
    if model_name:
        m = _model_manager.get_model(model_name)
        if m:
            default_system = m.system_prompt
    template = _prompt_library.get_or_default(template_name, default_system)
    return template.build_system()


def _attach_kb_note(system_prompt: str, kb_sources: list[str]) -> str:
    """Append a knowledge-base awareness note to a system prompt.

    Idempotent: strips any existing note before re-appending so the list
    stays current when new files are ingested mid-session.
    """
    _MARKER = "\n\n[Knowledge base:"
    base = system_prompt.split(_MARKER)[0]
    if not kb_sources:
        return base
    names = ", ".join(f'"{n}"' for n in kb_sources)
    return (
        base
        + f"{_MARKER} {names} — these documents are indexed and searchable. "
        "Use the retrieve_knowledge tool whenever the user asks about their content.]"
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

async def _collect_and_stream(gen, msg: cl.Message) -> None:
    """Drain an async generator, forwarding each token to a Chainlit message."""
    async for token in gen:
        await msg.stream_token(token)


def _build_validation_report(issues: list[ValidationIssue]) -> str | None:
    """Format validation issues as a markdown warning block.

    Returns None if there are no issues (clean config — no message shown).
    """
    if not issues:
        return None
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    lines = ["⚠ **Startup validation found configuration issues:**\n"]
    if errors:
        lines.append("**Errors** — affected models excluded from routing:")
        for i in errors:
            lines.append(f"- `{i.model_name}`: {i.message}")
    if warnings:
        if errors:
            lines.append("")
        lines.append("**Warnings** — models still available but may behave unexpectedly:")
        for i in warnings:
            lines.append(f"- `{i.model_name}`: {i.message}")
    lines.append("\nFix these in `configs/models.yaml` or the model directory, then restart Auri.")
    return "\n".join(lines)


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

    # 2. Run startup validation — exclude error-level models, surface warnings
    validation_issues = _model_manager.validate()
    if validation_issues:
        errors = [i for i in validation_issues if i.level == "error"]
        for issue in errors:
            model_cfg = _model_manager.get_model(issue.model_name)
            if model_cfg:
                model_cfg.available = False
                logger.error(
                    "Startup validation error for '%s': %s (marked unavailable)",
                    issue.model_name, issue.message,
                )
        for issue in [i for i in validation_issues if i.level == "warning"]:
            logger.warning(
                "Startup validation warning for '%s': %s",
                issue.model_name, issue.message,
            )
        report = _build_validation_report(validation_issues)
        if report:
            await cl.Message(content=report).send()

    # 3. Build lists for dropdowns
    model_values = _model_selector_values()
    lora_values = _lora_values()
    workspace_display_names = _workspace_manager.display_names()

    if len(model_values) <= 1:  # only "Auto" — no real models found
        await cl.Message(
            content=(
                "No models found. Add model directories to `models/vllm/` or `models/ollama/` "
                "and optionally configure them in `configs/models.yaml`."
            )
        ).send()
        return

    # 4. Send sidebar settings (workspace selector first)
    settings = await cl.ChatSettings(
        [
            Select(
                id="workspace",
                label="Workspace",
                values=workspace_display_names,
                initial_index=0,
            ),
            Select(
                id="model",
                label="Model",
                values=model_values,
                initial_index=0,
            ),
            Select(
                id="task_mode",
                label="Task Mode",
                values=_task_mode_loader.display_names(),
                initial_index=0,
            ),
            Select(
                id="lora",
                label="LoRA Adapter",
                values=lora_values,
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

    # 5. Initialise per-session state
    default_ws = _workspace_manager.ensure_default()
    registry, ingestor = _build_workspace_session(default_ws)

    system_prompt = _system_prompt_for(None, "Chat")
    cl.user_session.set("settings", settings)
    cl.user_session.set("message_history", [{"role": "system", "content": system_prompt}])
    cl.user_session.set("conversation_memory", ConversationMemory())
    cl.user_session.set("active_workspace", default_ws)
    cl.user_session.set("tool_registry", registry)
    cl.user_session.set("ingestor", ingestor)
    cl.user_session.set("kb_sources", [])

    # 6. Show offline warning if needed
    offline = [m.name for m in _model_manager.list_models() if not m.available]
    if offline:
        await cl.Message(
            content=f"Ollama is not running. The following models are offline: {', '.join(offline)}"
        ).send()

    logger.info("Chat session started (workspace=%s)", default_ws.name)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    old_settings = cl.user_session.get("settings") or {}
    old_model      = old_settings.get("model") if isinstance(old_settings, dict) else None
    old_mode       = old_settings.get("task_mode") if isinstance(old_settings, dict) else None
    old_workspace  = old_settings.get("workspace") if isinstance(old_settings, dict) else None

    new_model     = settings.get("model")
    new_mode      = settings.get("task_mode", "Chat")
    new_workspace = settings.get("workspace")

    cl.user_session.set("settings", settings)

    # ── Workspace switch ───────────────────────────────────────────────────────
    if new_workspace and new_workspace != old_workspace:
        ws = _workspace_manager.get_by_display_name(new_workspace)
        if ws is None:
            # Shouldn't happen (sidebar only shows existing workspaces), but guard
            ws = _workspace_manager.ensure_default()

        registry, ingestor = _build_workspace_session(ws)
        cl.user_session.set("active_workspace", ws)
        cl.user_session.set("tool_registry", registry)
        cl.user_session.set("ingestor", ingestor)
        cl.user_session.set("kb_sources", [])          # sources belong to old workspace
        cl.user_session.set("conversation_memory", ConversationMemory())  # fresh context

        # Replace history entirely — old turns belong to the previous workspace.
        # Only the fresh system prompt carries over; nothing from the old context.
        real_model = None if new_model == _AUTO_LABEL else new_model
        system_prompt = _system_prompt_for(real_model, new_mode or "Chat")
        cl.user_session.set("message_history", [{"role": "system", "content": system_prompt}])

        logger.info("Switched to workspace '%s'", ws.name)
        await cl.Message(content=f"Switched to workspace **{ws.display_name}**.").send()

    # ── Model or task mode change ──────────────────────────────────────────────
    if new_model != old_model or new_mode != old_mode:
        real_model = None if new_model == _AUTO_LABEL else new_model
        system_prompt = _system_prompt_for(real_model, new_mode or "Chat")
        kb_sources = cl.user_session.get("kb_sources") or []
        system_prompt = _attach_kb_note(system_prompt, kb_sources)
        history = cl.user_session.get("message_history") or []
        if history and history[0].get("role") == "system":
            history[0]["content"] = system_prompt
        cl.user_session.set("message_history", history)

    # ── Pre-warm vLLM if a specific vLLM model is selected ────────────────────
    if new_model and new_model != _AUTO_LABEL and new_model != old_model:
        new_model_cfg = _model_manager.get_model(new_model)
        if new_model_cfg and new_model_cfg.backend == "vllm":
            if not new_model_cfg.available:
                await cl.Message(
                    content=f"Model **{new_model_cfg.display_name}** is not available."
                ).send()
                return

            pinned = [
                _model_manager.get_lora(n)
                for n in new_model_cfg.pinned_loras
                if _model_manager.get_lora(n)
            ]

            status_msg = cl.Message(content=f"Loading **{new_model_cfg.display_name}**...")
            await status_msg.send()
            try:
                await _vllm_server.ensure_model(new_model_cfg, pinned)
                status_msg.content = f"**{new_model_cfg.display_name}** is ready."
            except Exception as exc:
                logger.error("Failed to load model '%s': %s", new_model, exc)
                status_msg.content = (
                    f"Failed to load **{new_model_cfg.display_name}**: {exc}\n\n"
                    "Check `logs/vllm_*.log` for details."
                )
            await status_msg.update()

    logger.debug(
        "Settings updated: workspace=%s model=%s task_mode=%s lora=%s",
        new_workspace, new_model, new_mode, settings.get("lora"),
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = cl.user_session.get("settings") or {}
    history: list[dict] = cl.user_session.get("message_history") or []
    tool_registry: ToolRegistry = cl.user_session.get("tool_registry")
    ingestor: Ingestor = cl.user_session.get("ingestor")
    active_workspace: Workspace = cl.user_session.get("active_workspace")

    # Resolve current sidebar selections
    model_setting: str = settings.get("model", _AUTO_LABEL) if isinstance(settings, dict) else _AUTO_LABEL
    task_mode: str     = settings.get("task_mode", "Chat") if isinstance(settings, dict) else "Chat"
    lora_name: str     = settings.get("lora", "None") if isinstance(settings, dict) else "None"
    temperature: float = float(settings.get("temperature", 0.7) if isinstance(settings, dict) else 0.7)

    # ── Workspace command: /workspace <name> creates a new workspace ───────────
    if message.content.strip().startswith("/workspace "):
        new_name = message.content.strip()[len("/workspace "):].strip()
        if new_name:
            ws = _workspace_manager.create_workspace(new_name)
            registry, ws_ingestor = _build_workspace_session(ws)
            cl.user_session.set("active_workspace", ws)
            cl.user_session.set("tool_registry", registry)
            cl.user_session.set("ingestor", ws_ingestor)
            cl.user_session.set("kb_sources", [])
            cl.user_session.set("conversation_memory", ConversationMemory())
            await cl.Message(
                content=f"Created and switched to workspace **{ws.display_name}**. "
                        f"Refresh the sidebar to select it."
            ).send()
        else:
            workspaces = _workspace_manager.list_workspaces()
            lines = [f"- **{ws.display_name}** (`{ws.name}`)" for ws in workspaces]
            await cl.Message(
                content="**Workspaces:**\n" + "\n".join(lines)
                + "\n\nTo create one: `/workspace <name>`"
            ).send()
        return

    # ── File upload: ingest documents into workspace knowledge base ────────────
    ingested_names: list[str] = []
    for el in message.elements or []:
        mime = getattr(el, "mime", "") or ""
        path = getattr(el, "path", None)
        name = getattr(el, "name", "") or ""
        if mime.startswith("image/") or not path:
            continue
        file_path = Path(path)
        name_suffix = Path(name).suffix.lower() if name else ""
        suffix = name_suffix or file_path.suffix.lower()
        if ingestor and suffix in ingestor.supported_suffixes:
            try:
                n = ingestor.ingest_upload(file_path, name)
                file_type = classify_file_type(Path(name))
                ingested_names.append(name)
                await cl.Message(
                    content=f"Ingested **{name}** ({file_type}) into knowledge base ({n} chunk{'s' if n != 1 else ''})."
                ).send()
                logger.info("Ingested file '%s' [%s] → %d chunks", name, file_type, n)
            except Exception as exc:
                logger.warning("Failed to ingest '%s': %s", name, exc)
                await cl.Message(content=f"Could not ingest **{name}**: {exc}").send()
        elif suffix:
            await cl.Message(
                content=f"**{name}** (`{suffix}`) is not a supported document type and was not ingested."
            ).send()
        else:
            await cl.Message(
                content=f"**{name}** could not be ingested (unknown file type)."
            ).send()

    # ── Persist KB sources and update system prompt after ingest ──────────────
    if ingested_names:
        kb_sources: list[str] = cl.user_session.get("kb_sources") or []
        kb_sources.extend(ingested_names)
        cl.user_session.set("kb_sources", kb_sources)
        if history and history[0].get("role") == "system":
            history[0]["content"] = _attach_kb_note(history[0]["content"], kb_sources)
            cl.user_session.set("message_history", history)

    # Detect attachments for intent signal
    has_image = any(
        getattr(el, "mime", "").startswith("image/")
        for el in (message.elements or [])
    )
    has_file = bool(message.elements) and not has_image

    # ── Intent detection ──────────────────────────────────────────────────────
    intent = _intent_classifier.classify(
        text=message.content,
        has_image=has_image,
        has_file=has_file,
    )

    # ── Effective task mode (needed before model selection for tool scoping) ──
    auto_routed = model_setting == _AUTO_LABEL
    effective_mode = (
        _INTENT_TO_MODE.get(intent.task, "Chat") if auto_routed else task_mode
    )

    # ── Resolve task mode and build scoped tool registry ─────────────────────
    mode_obj = _task_mode_loader.resolve(effective_mode)
    if mode_obj and mode_obj.enabled_tools and tool_registry:
        active_registry = tool_registry.scoped(mode_obj.enabled_tools)
    else:
        active_registry = tool_registry  # Chat: all auto-approved tools available

    # ── Model selection ───────────────────────────────────────────────────────
    if auto_routed:
        auto_model = _router.auto_select(
            intent,
            tool_registry=active_registry,
            message_text=message.content,
        )
        if auto_model is None:
            await cl.Message(
                content="No available models found. Please check Ollama is running or add models."
            ).send()
            return
        model_name = auto_model.name
        logger.info(
            "Auto-selected model '%s' (mode=%s) for intent '%s' (signals: %s)",
            model_name, effective_mode, intent.task, intent.signals,
        )
    else:
        model_name = model_setting

    model_cfg = _model_manager.get_model(model_name)
    if not model_cfg:
        await cl.Message(
            content=f"Model '{model_name}' not found. Please select a different model."
        ).send()
        return

    # ── Max tokens: task mode cap > template cap > sidebar value ─────────────
    sidebar_max_tokens: int = int(
        settings.get("max_tokens", model_cfg.max_tokens) if isinstance(settings, dict) else 2048
    )
    template_name = mode_obj.prompt_template if mode_obj else "chat"
    template = _prompt_library.get(template_name)
    mode_cap = mode_obj.max_output_tokens if mode_obj else None
    template_cap = template.max_output_tokens if template else None
    effective_cap = mode_cap or template_cap
    max_tokens = min(sidebar_max_tokens, effective_cap) if effective_cap else sidebar_max_tokens

    # ── System prompt for this turn ───────────────────────────────────────────
    if auto_routed:
        system_prompt = _system_prompt_for(model_name, effective_mode)
        if history and history[0].get("role") == "system":
            history = [{"role": "system", "content": system_prompt}] + history[1:]

    # ── LoRA validation ───────────────────────────────────────────────────────
    resolved_lora = lora_name if lora_name != "None" else None
    if resolved_lora:
        warning = _router.incompatible_lora_message(model_name, resolved_lora)
        if warning:
            await cl.Message(content=warning).send()
            resolved_lora = None

    # ── Append user turn to history ───────────────────────────────────────────
    user_content = message.content
    if ingested_names:
        names_str = ", ".join(f'"{n}"' for n in ingested_names)
        user_content += (
            f"\n\n[System note: The following file(s) were just ingested into the knowledge base: "
            f"{names_str}. Use the retrieve_knowledge tool to access their content.]"
        )
    history.append({"role": "user", "content": user_content})

    # ── Conversation memory: extract signals, inject context ─────────────────
    memory: ConversationMemory = cl.user_session.get("conversation_memory") or ConversationMemory()
    memory.active_task_mode = effective_mode
    pre_delta = _memory_extractor.extract_from_message(message.content, memory)

    # ── System text: base prompt + project memory + conversation memory ───────
    system_msg = history[0] if history and history[0].get("role") == "system" else None
    chat_history = history[1:] if system_msg else history
    system_text = system_msg["content"] if system_msg else "You are Auri, a helpful assistant."

    # Inject project memory (persistent workspace facts)
    project_memory = active_workspace.load_memory() if active_workspace else None
    project_facts_count = len(project_memory.facts) if project_memory else 0
    if project_memory:
        project_injection = project_memory.format_injection(active_workspace.display_name)
        if project_injection:
            system_text = system_text + "\n\n" + project_injection

    # Inject conversation memory (per-turn signals)
    conv_injection = memory.format_injection()
    if conv_injection:
        system_text = system_text + "\n\n" + conv_injection

    packed = _context_packer.pack(
        system_prompt=system_text,
        history=chat_history,
        context_limit=model_cfg.max_model_len,
        output_budget=max_tokens,
    )

    # ── Build RunContext ──────────────────────────────────────────────────────
    run_ctx = RunContext(
        intent=intent.task,
        task_mode=effective_mode,
        model_name=model_name,
        model_display_name=model_cfg.display_name,
        backend=model_cfg.backend,
        auto_routed=auto_routed,
        tools_available=active_registry.names() if active_registry else [],
        context_truncated=packed.truncated,
        workspace_name=active_workspace.name if active_workspace else "",
        project_facts_count=project_facts_count,
        memory_injected=memory.format_summary() if conv_injection else "",
    )

    # ── Stream response (with timeout, retry, and fallback) ───────────────────

    async def _attempt(m_name: str, m_lora, m_ctx: RunContext) -> bool:
        """One inference attempt with a hard timeout. Returns True on success."""
        try:
            await asyncio.wait_for(
                _collect_and_stream(
                    _router.route_request(
                        model_name=m_name,
                        lora_name=m_lora,
                        messages=packed.messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tool_registry=active_registry,
                        run_ctx=m_ctx,
                    ),
                    response_msg,
                ),
                timeout=_INFERENCE_TIMEOUT_S,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Inference timed out after %ds (model='%s')",
                _INFERENCE_TIMEOUT_S, m_name,
            )
            return False

    response_msg = cl.Message(content="")
    ok = False
    try:
        ok = await _attempt(model_name, resolved_lora, run_ctx)

        if not ok:
            await response_msg.update()
            await cl.Message(
                content=f"⏱ Response timed out after {_INFERENCE_TIMEOUT_S}s. Retrying…"
            ).send()
            run_ctx.fallback_reason = "timed out — retried same model"
            response_msg = cl.Message(content="")
            ok = await _attempt(model_name, resolved_lora, run_ctx)

        if not ok:
            await response_msg.update()
            fallback_cfg = _router.auto_select(intent, exclude={model_name}, tool_registry=active_registry)
            if fallback_cfg and fallback_cfg.name != model_name:
                await cl.Message(
                    content=f"⏱ Still timing out. Switching to **{fallback_cfg.display_name}**…"
                ).send()
                run_ctx.model_name = fallback_cfg.name
                run_ctx.model_display_name = fallback_cfg.display_name
                run_ctx.backend = fallback_cfg.backend
                run_ctx.fallback_reason = f"timed out — fell back to {fallback_cfg.name}"
                response_msg = cl.Message(content="")
                ok = await _attempt(fallback_cfg.name, None, run_ctx)

        if not ok:
            await cl.Message(
                content=(
                    "❌ Inference timed out on all retry attempts. "
                    "The model may be overloaded or unresponsive. Please try again later."
                )
            ).send()
    finally:
        await response_msg.update()

    # ── Post-inference memory update (sources from this run) ─────────────────
    post_delta = _memory_extractor.update_from_run(run_ctx, memory)
    all_updates = pre_delta.format_lines() + post_delta.format_lines()
    if all_updates:
        run_ctx.memory_updates = all_updates
    cl.user_session.set("conversation_memory", memory)

    # ── Record runtime metrics ────────────────────────────────────────────────
    _metrics_collector.record(run_ctx, success=ok)

    # ── Transparency panel ────────────────────────────────────────────────────
    await cl.Message(content=run_ctx.format_panel()).send()

    # ── Append assistant turn to history (full, not packed) ───────────────────
    if response_msg.content:
        history.append({"role": "assistant", "content": response_msg.content})
    cl.user_session.set("message_history", history)


@cl.on_chat_end
async def on_chat_end() -> None:
    _metrics_collector.log_summary()
    logger.debug("Chat session ended")
