"""
Tests for TaskModeLoader — YAML-driven task mode definitions.

Covers:
- load() from real YAML files (configs/task_modes/)
- All 5 shipped modes present
- resolve() by internal name and by display name
- enabled_tools scoping
- max_output_tokens parsed correctly (None when absent)
- Missing directory falls back to built-in Chat mode
- Malformed YAML file is skipped; other modes still load
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from auri.task_mode import TaskMode, TaskModeLoader


# ── Helpers ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
TASK_MODES_DIR = PROJECT_ROOT / "configs" / "task_modes"


# ── Real YAML files (shipped modes) ───────────────────────────────────────────

@pytest.fixture(scope="module")
def loader():
    tl = TaskModeLoader(TASK_MODES_DIR)
    tl.load()
    return tl


def test_all_shipped_modes_present(loader):
    names = {m.name for m in loader.list_modes()}
    assert {"chat", "code_review", "compare", "extract", "summarize"}.issubset(names)


def test_display_names_populated(loader):
    for mode in loader.list_modes():
        assert mode.display_name, f"Mode '{mode.name}' has empty display_name"


def test_resolve_by_internal_name(loader):
    mode = loader.resolve("chat")
    assert mode is not None
    assert mode.name == "chat"


def test_resolve_by_display_name(loader):
    # "Chat" is the display_name in chat.yaml
    mode = loader.resolve("Chat")
    assert mode is not None
    assert mode.name == "chat"


def test_resolve_unknown_returns_none(loader):
    assert loader.resolve("nonexistent_mode_xyz") is None


def test_code_review_has_filesystem_tool(loader):
    mode = loader.resolve("code_review")
    assert mode is not None
    assert "filesystem" in mode.enabled_tools


def test_summarize_has_max_output_tokens(loader):
    mode = loader.resolve("summarize")
    assert mode is not None
    assert mode.max_output_tokens is not None
    assert mode.max_output_tokens > 0


def test_chat_mode_has_no_token_cap(loader):
    mode = loader.resolve("chat")
    assert mode is not None
    assert mode.max_output_tokens is None


def test_extract_mode_tools_restricted(loader):
    mode = loader.resolve("extract")
    assert mode is not None
    # extract should not have terminal (confirm-required)
    assert "terminal" not in mode.enabled_tools


# ── Fallback behaviour ────────────────────────────────────────────────────────

def test_missing_directory_returns_chat_fallback(tmp_path):
    loader = TaskModeLoader(tmp_path / "nonexistent_dir")
    loader.load()
    modes = loader.list_modes()
    assert len(modes) == 1
    assert modes[0].name == "chat"


def test_empty_directory_returns_chat_fallback(tmp_path):
    loader = TaskModeLoader(tmp_path)  # empty dir
    loader.load()
    modes = loader.list_modes()
    assert len(modes) == 1
    assert modes[0].name == "chat"


# ── Synthetic YAML tests ──────────────────────────────────────────────────────

def test_load_synthetic_mode(tmp_path):
    yaml_content = {
        "display_name": "Test Mode",
        "description": "A test mode",
        "preferred_capabilities": ["coding"],
        "enabled_tools": ["filesystem", "git"],
        "prompt_template": "code_review",
        "response_format": "text",
        "max_output_tokens": 1234,
    }
    (tmp_path / "test_mode.yaml").write_text(yaml.dump(yaml_content))

    loader = TaskModeLoader(tmp_path)
    loader.load()

    mode = loader.resolve("test_mode")
    assert mode is not None
    assert mode.display_name == "Test Mode"
    assert mode.max_output_tokens == 1234
    assert mode.enabled_tools == ["filesystem", "git"]
    assert mode.preferred_capabilities == ["coding"]


def test_mode_without_max_tokens_is_none(tmp_path):
    yaml_content = {
        "display_name": "No Cap Mode",
        "prompt_template": "chat",
    }
    (tmp_path / "nocap.yaml").write_text(yaml.dump(yaml_content))

    loader = TaskModeLoader(tmp_path)
    loader.load()

    mode = loader.resolve("nocap")
    assert mode is not None
    assert mode.max_output_tokens is None


def test_malformed_yaml_skipped_others_load(tmp_path):
    # Good mode
    good = {"display_name": "Good", "prompt_template": "chat"}
    (tmp_path / "a_good.yaml").write_text(yaml.dump(good))

    # Malformed YAML — will fail to parse
    (tmp_path / "b_bad.yaml").write_text("display_name: [unclosed bracket\n")

    loader = TaskModeLoader(tmp_path)
    loader.load()  # should not raise

    assert loader.resolve("a_good") is not None
    assert loader.resolve("b_bad") is None


def test_display_names_list(loader):
    names = loader.display_names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert len(names) == len(loader.list_modes())
