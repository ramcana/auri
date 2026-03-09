"""
Tests for auri/workspace.py — Workspace, WorkspaceManager, ProjectMemory.

Covers:
  slugify / is_valid_slug:
    - alphanumeric slug is valid
    - single char is valid
    - slug with hyphens is valid
    - slug with spaces is invalid (slugify fixes it)
    - slug with uppercase is invalid (slugify lowercases)
    - empty string slugifies to "workspace"
    - long names are truncated to 50 chars

  WorkspaceManager:
    - ensure_default creates a workspace named "default"
    - ensure_default is idempotent (safe to call multiple times)
    - create_workspace builds files/ and knowledge/ subdirs
    - create_workspace derives display_name from slug if not given
    - create_workspace with existing name returns it (no error)
    - create_workspace with invalid slug auto-slugifies
    - get_workspace returns None for unknown name
    - get_workspace returns Workspace for existing name
    - list_workspaces returns all created workspaces sorted by name
    - display_names returns list of display-name strings
    - get_by_display_name finds by display name
    - get_by_display_name returns None for unknown name

  Workspace properties:
    - files_dir is <root>/files
    - knowledge_dir is <root>/knowledge
    - memory_path is <root>/memory.json
    - ensure_dirs creates missing subdirs

  ProjectMemory:
    - is_empty when fresh
    - set_fact adds a new fact and returns True
    - set_fact same value returns False (no change)
    - set_fact updated value returns True
    - get_fact retrieves by key
    - get_fact returns None for unknown key
    - remove_fact removes and returns True
    - remove_fact on missing key returns False
    - format_injection empty when no description or facts
    - format_injection includes workspace name in header
    - format_injection includes description
    - format_injection includes facts as key=value
    - format_injection caps at 8 facts
    - save + load round-trips schema_version, description, facts
    - load from non-existent path returns empty ProjectMemory
    - load from corrupt JSON returns empty ProjectMemory
    - Workspace.load_memory + save_memory round-trip
"""
from __future__ import annotations

import json
import pytest

from auri.workspace import (
    Fact,
    ProjectMemory,
    Workspace,
    WorkspaceManager,
    is_valid_slug,
    slugify,
)


# ── slugify / is_valid_slug ────────────────────────────────────────────────────

def test_valid_slug_alphanumeric():
    assert is_valid_slug("myproject")

def test_valid_slug_single_char():
    assert is_valid_slug("a")

def test_valid_slug_with_hyphens():
    assert is_valid_slug("my-project-v2")

def test_invalid_slug_spaces():
    assert not is_valid_slug("my project")

def test_invalid_slug_uppercase():
    assert not is_valid_slug("MyProject")

def test_invalid_slug_leading_hyphen():
    assert not is_valid_slug("-project")

def test_invalid_slug_trailing_hyphen():
    assert not is_valid_slug("project-")

def test_invalid_slug_empty():
    assert not is_valid_slug("")

def test_slugify_spaces():
    assert slugify("My Project") == "my-project"

def test_slugify_uppercase():
    assert slugify("MyProject") == "myproject"

def test_slugify_special_chars():
    assert slugify("project/v2!name") == "project-v2-name"

def test_slugify_empty_string():
    assert slugify("") == "workspace"

def test_slugify_long_name():
    result = slugify("a" * 100)
    assert len(result) <= 50

def test_slugify_strips_boundary_hyphens():
    result = slugify("  --project--  ")
    assert not result.startswith("-")
    assert not result.endswith("-")


# ── WorkspaceManager ──────────────────────────────────────────────────────────

@pytest.fixture
def manager(tmp_path):
    return WorkspaceManager(tmp_path / "workspaces")


def test_ensure_default_creates_workspace(manager):
    ws = manager.ensure_default()
    assert ws.name == "default"
    assert ws.root.is_dir()


def test_ensure_default_is_idempotent(manager):
    ws1 = manager.ensure_default()
    ws2 = manager.ensure_default()
    assert ws1.name == ws2.name
    assert ws1.root == ws2.root


def test_create_workspace_makes_subdirs(manager):
    ws = manager.create_workspace("test-proj")
    assert ws.files_dir.is_dir()
    assert ws.knowledge_dir.is_dir()


def test_create_workspace_derives_display_name(manager):
    ws = manager.create_workspace("my-project")
    assert ws.display_name == "My Project"


def test_create_workspace_uses_given_display_name(manager):
    ws = manager.create_workspace("my-project", "Custom Name")
    assert ws.display_name == "Custom Name"


def test_create_workspace_existing_name_idempotent(manager):
    ws1 = manager.create_workspace("alpha")
    ws2 = manager.create_workspace("alpha")
    assert ws1.root == ws2.root


def test_create_workspace_auto_slugifies_invalid_name(manager):
    ws = manager.create_workspace("My Cool Project!")
    assert is_valid_slug(ws.name)


def test_get_workspace_unknown_returns_none(manager):
    assert manager.get_workspace("does-not-exist") is None


def test_get_workspace_known_returns_workspace(manager):
    manager.create_workspace("beta")
    ws = manager.get_workspace("beta")
    assert ws is not None
    assert ws.name == "beta"


def test_list_workspaces_sorted(manager):
    manager.create_workspace("charlie")
    manager.create_workspace("alpha")
    manager.create_workspace("bravo")
    names = [ws.name for ws in manager.list_workspaces()]
    assert names == sorted(names)


def test_list_workspaces_all_created(manager):
    manager.create_workspace("one")
    manager.create_workspace("two")
    names = {ws.name for ws in manager.list_workspaces()}
    assert {"one", "two"}.issubset(names)


def test_display_names_returns_strings(manager):
    manager.create_workspace("proj-a")
    names = manager.display_names()
    assert all(isinstance(n, str) for n in names)
    assert any("Proj A" in n for n in names)


def test_get_by_display_name_found(manager):
    manager.create_workspace("gamma")
    ws = manager.get_by_display_name("Gamma")
    assert ws is not None
    assert ws.name == "gamma"


def test_get_by_display_name_not_found(manager):
    assert manager.get_by_display_name("Nonexistent") is None


# ── Workspace properties ───────────────────────────────────────────────────────

def test_workspace_files_dir(tmp_path):
    ws = Workspace(name="w", display_name="W", root=tmp_path / "w")
    assert ws.files_dir == tmp_path / "w" / "files"


def test_workspace_knowledge_dir(tmp_path):
    ws = Workspace(name="w", display_name="W", root=tmp_path / "w")
    assert ws.knowledge_dir == tmp_path / "w" / "knowledge"


def test_workspace_memory_path(tmp_path):
    ws = Workspace(name="w", display_name="W", root=tmp_path / "w")
    assert ws.memory_path == tmp_path / "w" / "memory.json"


def test_workspace_ensure_dirs_creates_them(tmp_path):
    ws = Workspace(name="w", display_name="W", root=tmp_path / "w")
    ws.ensure_dirs()
    assert ws.files_dir.is_dir()
    assert ws.knowledge_dir.is_dir()


# ── ProjectMemory ─────────────────────────────────────────────────────────────

def test_project_memory_is_empty_when_fresh():
    assert ProjectMemory().is_empty()


def test_set_fact_adds_new_fact_returns_true():
    m = ProjectMemory()
    assert m.set_fact("language", "Python") is True
    assert m.get_fact("language") == "Python"


def test_set_fact_same_value_returns_false():
    m = ProjectMemory()
    m.set_fact("language", "Python")
    assert m.set_fact("language", "Python") is False


def test_set_fact_updated_value_returns_true():
    m = ProjectMemory()
    m.set_fact("language", "Python")
    assert m.set_fact("language", "TypeScript") is True
    assert m.get_fact("language") == "TypeScript"


def test_set_fact_records_reason():
    m = ProjectMemory()
    m.set_fact("framework", "FastAPI", reason="detected in requirements.txt")
    assert m.facts[0].reason == "detected in requirements.txt"


def test_set_fact_records_turn():
    m = ProjectMemory()
    m.set_fact("env", "prod", turn=3)
    assert m.facts[0].updated_turn == 3


def test_get_fact_unknown_returns_none():
    m = ProjectMemory()
    assert m.get_fact("nonexistent") is None


def test_remove_fact_existing_returns_true():
    m = ProjectMemory()
    m.set_fact("key", "val")
    assert m.remove_fact("key") is True
    assert m.get_fact("key") is None


def test_remove_fact_missing_returns_false():
    m = ProjectMemory()
    assert m.remove_fact("ghost") is False


def test_multiple_facts_stored():
    m = ProjectMemory()
    m.set_fact("language", "Python")
    m.set_fact("framework", "FastAPI")
    assert len(m.facts) == 2


# ── ProjectMemory.format_injection ────────────────────────────────────────────

def test_format_injection_empty_when_no_data():
    assert ProjectMemory().format_injection() == ""


def test_format_injection_includes_workspace_name():
    m = ProjectMemory()
    m.description = "a REST API"
    inj = m.format_injection("my-project")
    assert "my-project" in inj


def test_format_injection_omits_workspace_name_if_empty():
    m = ProjectMemory()
    m.description = "a REST API"
    inj = m.format_injection()
    assert "Project context:" in inj


def test_format_injection_includes_description():
    m = ProjectMemory()
    m.description = "building a REST API backend"
    inj = m.format_injection("proj")
    assert "building a REST API backend" in inj


def test_format_injection_includes_facts():
    m = ProjectMemory()
    m.set_fact("language", "Python")
    m.set_fact("framework", "FastAPI")
    inj = m.format_injection("proj")
    assert "language=Python" in inj
    assert "framework=FastAPI" in inj


def test_format_injection_caps_at_8_facts():
    m = ProjectMemory()
    for i in range(12):
        m.set_fact(f"key{i}", f"val{i}")
    inj = m.format_injection("proj")
    # Only first 8 facts should appear
    assert "key7=val7" in inj
    assert "key8=val8" not in inj


def test_format_injection_description_only():
    m = ProjectMemory()
    m.description = "just a description"
    inj = m.format_injection()
    assert "just a description" in inj
    assert "=" not in inj  # no key=value lines


# ── ProjectMemory serialisation ───────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    m = ProjectMemory()
    m.description = "REST API project"
    m.set_fact("language", "Python", reason="detected", turn=2)
    path = tmp_path / "memory.json"
    m.save(path)

    loaded = ProjectMemory.load(path)
    assert loaded.description == "REST API project"
    assert loaded.get_fact("language") == "Python"
    assert loaded.facts[0].reason == "detected"
    assert loaded.facts[0].updated_turn == 2


def test_load_nonexistent_path_returns_empty(tmp_path):
    m = ProjectMemory.load(tmp_path / "ghost.json")
    assert m.is_empty()


def test_load_corrupt_json_returns_empty(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all {{", encoding="utf-8")
    m = ProjectMemory.load(bad)
    assert m.is_empty()


def test_schema_version_preserved(tmp_path):
    m = ProjectMemory()
    m.schema_version = 2
    path = tmp_path / "memory.json"
    m.save(path)
    loaded = ProjectMemory.load(path)
    assert loaded.schema_version == 2


# ── Workspace.load_memory / save_memory ───────────────────────────────────────

def test_workspace_save_and_load_memory(tmp_path):
    ws = Workspace(name="w", display_name="W", root=tmp_path / "w")
    ws.ensure_dirs()

    pm = ProjectMemory()
    pm.description = "test project"
    pm.set_fact("env", "dev")
    ws.save_memory(pm)

    loaded = ws.load_memory()
    assert loaded.description == "test project"
    assert loaded.get_fact("env") == "dev"


def test_workspace_load_memory_missing_returns_empty(tmp_path):
    ws = Workspace(name="w", display_name="W", root=tmp_path / "w")
    ws.ensure_dirs()
    pm = ws.load_memory()
    assert pm.is_empty()


# ── Display name persistence (meta.json) ──────────────────────────────────────

def test_create_workspace_persists_display_name(manager):
    ws = manager.create_workspace("my-proj", "My Custom Label")
    # Re-load via get_workspace — must return persisted name, not derived one
    ws2 = manager.get_workspace("my-proj")
    assert ws2.display_name == "My Custom Label"


def test_create_workspace_existing_keeps_original_display_name(manager):
    manager.create_workspace("dupe", "Original Name")
    ws2 = manager.create_workspace("dupe", "Different Name")  # second call, already exists
    assert ws2.display_name == "Original Name"


def test_list_workspaces_uses_persisted_display_names(manager):
    manager.create_workspace("tagged", "Tagged Project")
    names = {ws.display_name for ws in manager.list_workspaces()}
    assert "Tagged Project" in names


def test_display_name_survives_manager_recreation(tmp_path):
    # First manager instance creates a workspace
    mgr1 = WorkspaceManager(tmp_path / "ws")
    mgr1.create_workspace("persistent", "My Persistent Label")

    # Second manager instance in the same directory reads from meta.json
    mgr2 = WorkspaceManager(tmp_path / "ws")
    ws = mgr2.get_workspace("persistent")
    assert ws is not None
    assert ws.display_name == "My Persistent Label"


def test_slugified_name_stores_original_display_name(manager):
    # "My Cool Project!" is auto-slugified → "my-cool-project"
    ws = manager.create_workspace("My Cool Project!", "My Cool Project!")
    # Slug should be valid
    assert is_valid_slug(ws.name)
    # Display name as provided should survive
    ws2 = manager.get_workspace(ws.name)
    assert ws2.display_name == "My Cool Project!"


# ── Slug collision detection ───────────────────────────────────────────────────

def test_slug_collision_same_directory(manager):
    # "My Project" and "my project" both slug to "my-project" → same workspace
    ws1 = manager.create_workspace("my project", "My Project")
    ws2 = manager.create_workspace("my-project")   # same slug, different call
    assert ws1.root == ws2.root


def test_slug_collision_first_display_name_wins(manager):
    manager.create_workspace("my project", "First Label")
    ws2 = manager.create_workspace("my-project", "Second Label")
    assert ws2.display_name == "First Label"


# ── Structural isolation between workspaces ───────────────────────────────────

def test_two_workspaces_have_different_knowledge_dirs(manager):
    ws_a = manager.create_workspace("alpha")
    ws_b = manager.create_workspace("beta")
    assert ws_a.knowledge_dir != ws_b.knowledge_dir


def test_two_workspaces_have_different_files_dirs(manager):
    ws_a = manager.create_workspace("alpha")
    ws_b = manager.create_workspace("beta")
    assert ws_a.files_dir != ws_b.files_dir


def test_workspace_project_memories_are_independent(manager):
    ws_a = manager.create_workspace("ws-a")
    ws_b = manager.create_workspace("ws-b")

    pm_a = ProjectMemory()
    pm_a.set_fact("language", "Python")
    ws_a.save_memory(pm_a)

    pm_b = ProjectMemory()
    pm_b.set_fact("language", "TypeScript")
    ws_b.save_memory(pm_b)

    # Each workspace reads its own facts
    assert ws_a.load_memory().get_fact("language") == "Python"
    assert ws_b.load_memory().get_fact("language") == "TypeScript"


def test_workspace_memory_not_shared_between_instances(manager):
    ws_a = manager.create_workspace("isolated-a")
    ws_b = manager.create_workspace("isolated-b")

    pm = ProjectMemory()
    pm.description = "only in A"
    ws_a.save_memory(pm)

    assert ws_b.load_memory().is_empty()


# ── RunContext workspace panel row ────────────────────────────────────────────

def test_run_context_workspace_row_shown():
    import time
    from auri.run_context import RunContext
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    ctx.workspace_name = "my-project"
    panel = ctx.format_panel()
    assert "Workspace" in panel
    assert "my-project" in panel


def test_run_context_workspace_shows_fact_count():
    import time
    from auri.run_context import RunContext
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    ctx.workspace_name = "proj"
    ctx.project_facts_count = 3
    panel = ctx.format_panel()
    assert "3 project facts" in panel


def test_run_context_workspace_singular_fact():
    import time
    from auri.run_context import RunContext
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    ctx.workspace_name = "proj"
    ctx.project_facts_count = 1
    panel = ctx.format_panel()
    assert "1 project fact" in panel
    assert "facts" not in panel


def test_run_context_no_workspace_row_when_empty():
    import time
    from auri.run_context import RunContext
    ctx = RunContext(model_name="m", model_display_name="M", backend="ollama")
    ctx._start = time.monotonic() - 0.1
    panel = ctx.format_panel()
    assert "Workspace" not in panel
