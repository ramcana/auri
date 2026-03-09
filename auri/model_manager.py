"""
ModelManager: filesystem scanner + YAML config merger.

Discovery rules:
  models/vllm/<name>/   — valid if contains config.json OR any .safetensors / .bin file
  models/ollama/<name>/ — always valid; model.txt inside overrides the Ollama tag
  Ollama daemon API     — GET /api/tags auto-discovers all `ollama pull`ed models
  loras/<name>/         — valid if contains adapter_config.json

Override precedence (highest to lowest):
  YAML per-model entry > YAML defaults > Ollama auto-discovery > Python dataclass defaults
Filesystem wins on existence for vLLM; Ollama daemon is the source of truth for Ollama models.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

from auri.settings import AppSettings

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """One validation finding for a model or LoRA."""
    model_name: str
    level: str       # "error" | "warning"
    message: str


@dataclass
class LoRAConfig:
    name: str
    path: Path
    display_name: str
    base_model: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    backend: Literal["vllm", "ollama"]
    display_name: str
    available: bool = True           # set False when Ollama is offline

    # vLLM fields (ignored for ollama backend)
    path: Optional[Path] = None
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    max_tokens: int = 2048
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    extra_vllm_args: list[str] = field(default_factory=list)
    compatible_loras: list[str] = field(default_factory=list)
    pinned_loras: list[str] = field(default_factory=list)

    # Ollama fields (ignored for vllm backend)
    ollama_model_name: Optional[str] = None

    # Capabilities used for intent-based routing
    # Values: "chat", "coding", "tools", "vision"
    capabilities: list[str] = field(default_factory=list)

    # Shared
    system_prompt: str = "You are Auri, a helpful assistant."
    temperature: float = 0.7


# ── ModelManager ──────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._models: dict[str, ModelConfig] = {}
        self._loras: dict[str, LoRAConfig] = {}
        self._yaml_defaults: dict = {}
        self._lock = threading.Lock()  # guards reload() called from watchdog thread

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Parse YAML config, scan filesystem, apply overrides, validate cross-references."""
        raw = self._load_yaml()
        self._yaml_defaults = raw.get("defaults", {})

        models: dict[str, ModelConfig] = {}
        loras: dict[str, LoRAConfig] = {}

        self._scan_vllm_models(models)
        self._scan_ollama_models(models)
        self._scan_loras(loras)
        self._apply_yaml_overrides(models, loras, raw)   # YAML runs first so its tags get claimed
        self._scan_ollama_daemon(models)  # daemon fills in any remaining pulled models
        self._validate_references(models, loras)

        with self._lock:
            self._models = models
            self._loras = loras

        logger.info(
            "ModelManager loaded %d model(s) and %d LoRA(s)",
            len(models),
            len(loras),
        )

    def reload(self) -> None:
        """Re-scan filesystem and re-merge YAML (thread-safe, for watchdog / manual refresh)."""
        logger.info("ModelManager: reloading...")
        self.load()

    def list_models(self) -> list[ModelConfig]:
        with self._lock:
            return sorted(self._models.values(), key=lambda m: m.display_name)

    def list_model_names(self) -> list[str]:
        return [m.name for m in self.list_models()]

    def get_model(self, name: str) -> Optional[ModelConfig]:
        with self._lock:
            return self._models.get(name)

    def list_loras(self) -> list[LoRAConfig]:
        with self._lock:
            return sorted(self._loras.values(), key=lambda l: l.display_name)

    def list_lora_names(self) -> list[str]:
        return [l.name for l in self.list_loras()]

    def get_lora(self, name: str) -> Optional[LoRAConfig]:
        with self._lock:
            return self._loras.get(name)

    def get_compatible_loras(self, model_name: str) -> list[LoRAConfig]:
        """Return LoRAs listed in model's compatible_loras that exist on disk."""
        model = self.get_model(model_name)
        if model is None or not model.compatible_loras:
            return []
        result = []
        for lora_name in model.compatible_loras:
            lora = self.get_lora(lora_name)
            if lora:
                result.append(lora)
        return result

    # ── Capability validation ─────────────────────────────────────────────────

    _VALID_CAPABILITIES: frozenset[str] = frozenset({"chat", "coding", "tools", "vision"})

    def validate(self) -> list[ValidationIssue]:
        """Run config validation on all loaded models and LoRAs.

        Returns a list of ValidationIssues (level='error' or 'warning').
        Errors indicate the model will likely fail at inference time.
        Warnings indicate suspicious config that may cause unexpected behaviour.

        Does NOT mutate model state — callers decide how to act on results.
        """
        issues: list[ValidationIssue] = []
        with self._lock:
            models = dict(self._models)
            loras = dict(self._loras)
        for model in models.values():
            issues.extend(self._validate_model(model, loras))
        return issues

    def _validate_model(
        self,
        model: ModelConfig,
        loras: dict[str, LoRAConfig],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # 1. vLLM path existence and weight files
        if model.backend == "vllm":
            if model.path is None:
                issues.append(ValidationIssue(model.name, "error", "no model path configured"))
            elif not model.path.exists():
                issues.append(ValidationIssue(
                    model.name, "error",
                    f"model directory does not exist: {model.path}",
                ))
            elif not (
                (model.path / "config.json").exists()
                or any(model.path.glob("*.safetensors"))
                or any(model.path.glob("*.bin"))
            ):
                issues.append(ValidationIssue(
                    model.name, "warning",
                    f"model directory has no config.json or weight files: {model.path}",
                ))

        # 2. Capability tags
        for cap in model.capabilities:
            if cap not in self._VALID_CAPABILITIES:
                issues.append(ValidationIssue(
                    model.name, "warning",
                    f"unknown capability '{cap}' "
                    f"(valid: {', '.join(sorted(self._VALID_CAPABILITIES))})",
                ))

        # 3. max_tokens vs max_model_len (vLLM only)
        if model.backend == "vllm" and model.max_tokens > model.max_model_len:
            issues.append(ValidationIssue(
                model.name, "warning",
                f"max_tokens ({model.max_tokens}) > max_model_len ({model.max_model_len}); "
                "generation may be silently capped by vLLM",
            ))

        # 4. gpu_memory_utilization range (vLLM only)
        if model.backend == "vllm" and not (0.0 < model.gpu_memory_utilization <= 1.0):
            issues.append(ValidationIssue(
                model.name, "error",
                f"gpu_memory_utilization {model.gpu_memory_utilization} is out of range (0.0, 1.0]",
            ))

        # 5. tensor_parallel_size (vLLM only)
        if model.backend == "vllm" and model.tensor_parallel_size < 1:
            issues.append(ValidationIssue(
                model.name, "error",
                f"tensor_parallel_size {model.tensor_parallel_size} must be ≥ 1",
            ))

        # 6. LoRA cross-references
        for lora_name in model.compatible_loras:
            if lora_name not in loras:
                issues.append(ValidationIssue(
                    model.name, "warning",
                    f"compatible_loras lists '{lora_name}' which was not found on disk",
                ))
        for lora_name in model.pinned_loras:
            if lora_name not in model.compatible_loras:
                issues.append(ValidationIssue(
                    model.name, "warning",
                    f"pinned_loras lists '{lora_name}' which is not in compatible_loras",
                ))

        return issues

    def mark_ollama_unavailable(self) -> None:
        """Mark all ollama-backend models as unavailable (Ollama offline)."""
        with self._lock:
            for model in self._models.values():
                if model.backend == "ollama":
                    model.available = False
        logger.warning("Ollama unavailable — all Ollama-backend models marked offline")

    def mark_ollama_available(self) -> None:
        with self._lock:
            for model in self._models.values():
                if model.backend == "ollama":
                    model.available = True

    # ── Filesystem scanning ───────────────────────────────────────────────────

    def _scan_vllm_models(self, models: dict[str, ModelConfig]) -> None:
        vllm_dir = self._settings.models_vllm_dir
        if not vllm_dir.is_dir():
            return
        for entry in vllm_dir.iterdir():
            if not entry.is_dir():
                continue
            # Valid model dir: has config.json OR at least one .safetensors / .bin file
            has_config = (entry / "config.json").exists()
            has_weights = any(
                entry.glob("*.safetensors")
            ) or any(entry.glob("*.bin"))
            if not (has_config or has_weights):
                logger.debug("Skipping %s — no config.json or weight files", entry)
                continue
            name = entry.name
            models[name] = ModelConfig(
                name=name,
                backend="vllm",
                display_name=name,
                path=entry.resolve(),
            )
            logger.debug("Discovered vLLM model: %s", name)

    def _scan_ollama_models(self, models: dict[str, ModelConfig]) -> None:
        ollama_dir = self._settings.models_ollama_dir
        if not ollama_dir.is_dir():
            return
        for entry in ollama_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            # model.txt overrides the Ollama tag (handles colons in tags like "phi3:mini")
            model_txt = entry / "model.txt"
            ollama_tag = model_txt.read_text().strip() if model_txt.exists() else name
            models[name] = ModelConfig(
                name=name,
                backend="ollama",
                display_name=name,
                ollama_model_name=ollama_tag,
            )
            logger.debug("Discovered Ollama model: %s (tag: %s)", name, ollama_tag)

    def _scan_ollama_daemon(self, models: dict[str, ModelConfig]) -> None:
        """Query the Ollama daemon's /api/tags to auto-discover all pulled models.

        Any model already discovered (from models/ollama/ dirs) is skipped.
        New models are registered with a safe key derived from the tag
        (colons and dots replaced with hyphens, e.g. qwen2.5:7b → qwen2-5-7b).
        YAML overrides applied later can still customise display_name etc.
        """
        ollama_base = self._settings.ollama_base_url.rstrip("/")
        if ollama_base.endswith("/v1"):
            native_root = ollama_base[: -len("/v1")]
        else:
            native_root = ollama_base
        tags_url = f"{native_root}/api/tags"

        try:
            req = urllib.request.Request(tags_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.debug("Ollama daemon not reachable at %s — skipping auto-discovery: %s", tags_url, exc)
            return

        # Build a set of ollama tags already claimed by existing entries
        claimed_tags = {
            m.ollama_model_name
            for m in models.values()
            if m.backend == "ollama" and m.ollama_model_name
        }

        for entry in data.get("models", []):
            tag = entry.get("name", "")
            if not tag:
                continue
            if tag in claimed_tags:
                # Already covered by a filesystem dir or YAML entry
                continue
            key = re.sub(r"[:.]+", "-", tag).strip("-")
            if key in models:
                continue
            models[key] = ModelConfig(
                name=key,
                backend="ollama",
                display_name=tag,
                ollama_model_name=tag,
            )
            claimed_tags.add(tag)
            logger.debug("Auto-discovered Ollama model from daemon: %s (key: %s)", tag, key)

        logger.info(
            "Ollama daemon scan complete — %d total Ollama model(s) registered",
            sum(1 for m in models.values() if m.backend == "ollama"),
        )

    def _scan_loras(self, loras: dict[str, LoRAConfig]) -> None:
        loras_dir = self._settings.loras_dir
        if not loras_dir.is_dir():
            return
        for entry in loras_dir.iterdir():
            if not entry.is_dir():
                continue
            # Valid LoRA dir must contain adapter_config.json (HuggingFace PEFT marker)
            if not (entry / "adapter_config.json").exists():
                logger.debug("Skipping %s — no adapter_config.json", entry)
                continue
            name = entry.name
            loras[name] = LoRAConfig(
                name=name,
                path=entry.resolve(),
                display_name=name,
            )
            logger.debug("Discovered LoRA: %s", name)

    # ── YAML override application ─────────────────────────────────────────────

    def _apply_yaml_overrides(
        self,
        models: dict[str, ModelConfig],
        loras: dict[str, LoRAConfig],
        raw: dict,
    ) -> None:
        defaults = raw.get("defaults", {})
        yaml_models = raw.get("models") or {}
        yaml_loras = raw.get("loras") or {}

        for name, entry in yaml_models.items():
            if entry is None:
                entry = {}
            merged = {**defaults, **entry}

            if name not in models:
                # vLLM models require weights on disk — skip with a warning.
                # Ollama models are managed by Ollama itself, not our filesystem,
                # so a YAML-only entry is valid and creates the model directly.
                backend = entry.get("backend") or defaults.get("backend", "")
                if backend != "ollama":
                    logger.warning(
                        "YAML references model '%s' (backend=%s) but it was not found on disk — skipping",
                        name,
                        backend,
                    )
                    continue
                # Create Ollama model from YAML declaration alone
                ollama_tag = entry.get("ollama_model_name", name)
                models[name] = ModelConfig(
                    name=name,
                    backend="ollama",
                    display_name=entry.get("display_name", name),
                    ollama_model_name=ollama_tag,
                )
                logger.debug("Registered YAML-declared Ollama model: %s (tag: %s)", name, ollama_tag)

            model = models[name]
            # Apply YAML defaults first, then per-model entry (per-model wins)
            self._apply_model_fields(model, merged)

        for name, entry in yaml_loras.items():
            if entry is None:
                entry = {}
            if name not in loras:
                logger.warning(
                    "YAML references LoRA '%s' but it was not found on disk — skipping",
                    name,
                )
                continue
            lora = loras[name]
            if "display_name" in entry:
                lora.display_name = entry["display_name"]
            if "base_model" in entry:
                lora.base_model = entry["base_model"]
            if "path" in entry:
                # Allow YAML to specify an explicit path (relative to project root)
                p = Path(entry["path"])
                if not p.is_absolute():
                    p = self._settings.project_root / p
                lora.path = p.resolve()

    def _apply_model_fields(self, model: ModelConfig, entry: dict) -> None:
        """Apply YAML entry fields to an existing ModelConfig, field by field."""
        str_fields = {"display_name", "dtype", "system_prompt", "ollama_model_name"}
        float_fields = {"gpu_memory_utilization", "temperature"}
        int_fields = {"max_model_len", "max_tokens", "tensor_parallel_size"}
        list_str_fields = {"extra_vllm_args", "compatible_loras", "pinned_loras", "capabilities"}

        for f in str_fields:
            if f in entry and entry[f] is not None:
                setattr(model, f, str(entry[f]))
        for f in float_fields:
            if f in entry and entry[f] is not None:
                setattr(model, f, float(entry[f]))
        for f in int_fields:
            if f in entry and entry[f] is not None:
                setattr(model, f, int(entry[f]))
        for f in list_str_fields:
            if f in entry and entry[f] is not None:
                setattr(model, f, [str(x) for x in entry[f]])
        if "path" in entry and entry["path"] is not None:
            p = Path(entry["path"])
            if not p.is_absolute():
                p = self._settings.project_root / p
            model.path = p.resolve()
        if "backend" in entry and entry["backend"] in ("vllm", "ollama"):
            model.backend = entry["backend"]

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_references(
        self,
        models: dict[str, ModelConfig],
        loras: dict[str, LoRAConfig],
    ) -> None:
        for model in models.values():
            for lora_name in model.compatible_loras:
                if lora_name not in loras:
                    logger.warning(
                        "Model '%s' lists compatible LoRA '%s' but it was not found on disk",
                        model.name,
                        lora_name,
                    )
            for lora_name in model.pinned_loras:
                if lora_name not in model.compatible_loras:
                    logger.warning(
                        "Model '%s' has pinned LoRA '%s' not in compatible_loras",
                        model.name,
                        lora_name,
                    )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_yaml(self) -> dict:
        config_path = self._settings.config_path
        if not config_path.exists():
            logger.warning("No models.yaml found at %s — using filesystem discovery only", config_path)
            return {}
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}

        version = raw.get("schema_version")
        if version != 1:
            logger.warning(
                "models.yaml schema_version is %s (expected 1) — proceeding anyway",
                version,
            )
        return raw
