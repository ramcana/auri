"""
Tests for ModelManager.validate() — startup capability validation.

All tests bypass filesystem scanning by populating _models/_loras directly.
No real model files required.

Covers:
- vLLM model with missing path → error
- vLLM model path exists but no weights/config → warning
- Unknown capability tag → warning
- max_tokens > max_model_len → warning
- gpu_memory_utilization out of range → error
- tensor_parallel_size < 1 → error
- compatible_loras references unknown LoRA → warning
- pinned_loras not in compatible_loras → warning
- Clean model → no issues
- Ollama model (backend="ollama") skips vLLM-only checks
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from auri.model_manager import LoRAConfig, ModelConfig, ModelManager, ValidationIssue
from auri.settings import AppSettings


def make_manager() -> ModelManager:
    settings = MagicMock(spec=AppSettings)
    settings.project_root = Path("/tmp")
    manager = ModelManager(settings)
    manager._models = {}
    manager._loras = {}
    return manager


def vllm_model(name: str = "test-model", **kwargs) -> ModelConfig:
    defaults = dict(
        name=name,
        backend="vllm",
        display_name=name,
        path=None,
        capabilities=["chat"],
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        max_tokens=2048,
        tensor_parallel_size=1,
        compatible_loras=[],
        pinned_loras=[],
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def ollama_model(name: str = "llama3", **kwargs) -> ModelConfig:
    defaults = dict(
        name=name,
        backend="ollama",
        display_name=name,
        capabilities=["chat"],
        compatible_loras=[],
        pinned_loras=[],
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def fake_lora(name: str) -> LoRAConfig:
    return LoRAConfig(name=name, path=Path(f"/tmp/{name}"), display_name=name)


# ── vLLM path checks ──────────────────────────────────────────────────────────

def test_vllm_no_path_is_error():
    manager = make_manager()
    model = vllm_model(path=None)
    manager._models = {model.name: model}
    issues = manager.validate()
    errors = [i for i in issues if i.level == "error"]
    assert any("no model path" in i.message for i in errors)


def test_vllm_nonexistent_path_is_error(tmp_path):
    manager = make_manager()
    model = vllm_model(path=tmp_path / "does_not_exist")
    manager._models = {model.name: model}
    issues = manager.validate()
    errors = [i for i in issues if i.level == "error"]
    assert any("does not exist" in i.message for i in errors)


def test_vllm_path_no_weights_is_warning(tmp_path):
    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    # Directory exists but has no config.json or .safetensors/.bin
    manager = make_manager()
    model = vllm_model(path=model_dir)
    manager._models = {model.name: model}
    issues = manager.validate()
    warnings = [i for i in issues if i.level == "warning"]
    assert any("no config.json" in i.message for i in warnings)


def test_vllm_path_with_config_json_is_clean(tmp_path):
    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir)
    manager._models = {model.name: model}
    issues = manager.validate()
    path_issues = [i for i in issues if "config" in i.message or "path" in i.message.lower()]
    assert not path_issues


def test_vllm_path_with_safetensors_is_clean(tmp_path):
    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("fake")
    manager = make_manager()
    model = vllm_model(path=model_dir)
    manager._models = {model.name: model}
    issues = manager.validate()
    weight_issues = [i for i in issues if "weight" in i.message]
    assert not weight_issues


# ── Capability tags ───────────────────────────────────────────────────────────

def test_unknown_capability_is_warning(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, capabilities=["chat", "telepathy"])
    manager._models = {model.name: model}
    issues = manager.validate()
    warnings = [i for i in issues if i.level == "warning"]
    assert any("telepathy" in i.message for i in warnings)


def test_all_valid_capabilities_no_warning(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, capabilities=["chat", "coding", "tools", "vision"])
    manager._models = {model.name: model}
    issues = manager.validate()
    cap_warnings = [i for i in issues if "capability" in i.message.lower() or "unknown" in i.message]
    assert not cap_warnings


# ── Numeric constraints ────────────────────────────────────────────────────────

def test_max_tokens_exceeds_max_model_len_is_warning(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, max_tokens=8192, max_model_len=4096)
    manager._models = {model.name: model}
    issues = manager.validate()
    warnings = [i for i in issues if i.level == "warning"]
    assert any("max_tokens" in i.message for i in warnings)


def test_gpu_memory_utilization_zero_is_error(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, gpu_memory_utilization=0.0)
    manager._models = {model.name: model}
    issues = manager.validate()
    errors = [i for i in issues if i.level == "error"]
    assert any("gpu_memory_utilization" in i.message for i in errors)


def test_gpu_memory_utilization_above_one_is_error(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, gpu_memory_utilization=1.5)
    manager._models = {model.name: model}
    issues = manager.validate()
    errors = [i for i in issues if i.level == "error"]
    assert any("gpu_memory_utilization" in i.message for i in errors)


def test_valid_gpu_memory_utilization_no_error(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, gpu_memory_utilization=1.0)
    manager._models = {model.name: model}
    issues = manager.validate()
    gpu_errors = [i for i in issues if "gpu_memory" in i.message]
    assert not gpu_errors


def test_tensor_parallel_size_zero_is_error(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, tensor_parallel_size=0)
    manager._models = {model.name: model}
    issues = manager.validate()
    errors = [i for i in issues if i.level == "error"]
    assert any("tensor_parallel_size" in i.message for i in errors)


# ── LoRA cross-references ──────────────────────────────────────────────────────

def test_compatible_lora_not_on_disk_is_warning(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manager = make_manager()
    model = vllm_model(path=model_dir, compatible_loras=["ghost-lora"])
    manager._models = {model.name: model}
    manager._loras = {}  # ghost-lora not registered
    issues = manager.validate()
    warnings = [i for i in issues if i.level == "warning"]
    assert any("ghost-lora" in i.message for i in warnings)


def test_pinned_lora_not_in_compatible_is_warning(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    lora = fake_lora("my-lora")
    manager = make_manager()
    model = vllm_model(
        path=model_dir,
        compatible_loras=["my-lora"],
        pinned_loras=["other-lora"],  # not in compatible_loras
    )
    manager._models = {model.name: model}
    manager._loras = {"my-lora": lora}
    issues = manager.validate()
    warnings = [i for i in issues if i.level == "warning"]
    assert any("other-lora" in i.message for i in warnings)


# ── Ollama bypass ─────────────────────────────────────────────────────────────

def test_ollama_skips_vllm_checks():
    manager = make_manager()
    model = ollama_model(capabilities=["chat"])
    manager._models = {model.name: model}
    issues = manager.validate()
    # Ollama models have no path → but no error should be raised
    vllm_errors = [i for i in issues if "path" in i.message or "gpu_memory" in i.message]
    assert not vllm_errors


# ── Clean model ───────────────────────────────────────────────────────────────

def test_clean_vllm_model_has_no_issues(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    lora = fake_lora("lora-a")
    manager = make_manager()
    model = vllm_model(
        path=model_dir,
        capabilities=["chat", "coding"],
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        max_tokens=2048,
        tensor_parallel_size=1,
        compatible_loras=["lora-a"],
        pinned_loras=["lora-a"],
    )
    manager._models = {model.name: model}
    manager._loras = {"lora-a": lora}
    issues = manager.validate()
    assert issues == []
