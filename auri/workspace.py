"""
Project workspaces: named, isolated contexts with their own RAG index,
file storage, and persistent project memory.

Layout per workspace:
  workspaces/<name>/
    files/       — user-uploaded and generated files
    knowledge/   — ChromaDB RAG index for this workspace
    memory.json  — project-scoped persistent facts

WorkspaceManager keeps a single shared Embedder and creates per-workspace
VectorStore/Ingestor/Retriever instances on demand.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_META_FILENAME = "meta.json"

# ── Name validation ────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,48}[a-z0-9])?$")
_SLUGIFY_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Convert any string to a valid workspace slug (lowercase, hyphens only)."""
    slug = _SLUGIFY_STRIP.sub("-", name.lower().strip()).strip("-")
    return slug[:50] or "workspace"


def is_valid_slug(name: str) -> bool:
    return bool(_SLUG_RE.match(name))


# ── Project memory ─────────────────────────────────────────────────────────────

@dataclass
class Fact:
    key: str
    value: str
    reason: str = ""
    updated_turn: int = 0


@dataclass
class ProjectMemory:
    """Persistent, project-scoped facts stored as memory.json inside a workspace.

    Facts are free-form key-value pairs (e.g. language=Python, framework=FastAPI).
    The description field is a single prose sentence about the project.
    Both are injected into the system prompt as a compact block when non-empty.
    """

    schema_version: int = 1
    description: str = ""
    facts: list[Fact] = field(default_factory=list)

    _MAX_INJECT_FACTS = 8   # plain int, not a dataclass field

    def set_fact(self, key: str, value: str, reason: str = "", turn: int = 0) -> bool:
        """Add or update a fact. Returns True if the value actually changed."""
        for f in self.facts:
            if f.key == key:
                if f.value == value:
                    return False
                f.value = value
                f.reason = reason
                f.updated_turn = turn
                return True
        self.facts.append(Fact(key=key, value=value, reason=reason, updated_turn=turn))
        return True

    def get_fact(self, key: str) -> Optional[str]:
        for f in self.facts:
            if f.key == key:
                return f.value
        return None

    def remove_fact(self, key: str) -> bool:
        before = len(self.facts)
        self.facts = [f for f in self.facts if f.key != key]
        return len(self.facts) < before

    def is_empty(self) -> bool:
        return not self.description and not self.facts

    def format_injection(self, workspace_name: str = "") -> str:
        """Compact project context block for injection into the system prompt.

        Example output:
          Project context (my-project):
            Building a REST API backend
            language=Python  framework=FastAPI
        """
        if self.is_empty():
            return ""
        header = f"Project context ({workspace_name}):" if workspace_name else "Project context:"
        lines = [header]
        if self.description:
            lines.append(f"  {self.description}")
        for f in self.facts[:self._MAX_INJECT_FACTS]:
            lines.append(f"  {f.key}={f.value}")
        return "\n".join(lines)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "reason": f.reason,
                    "updated_turn": f.updated_turn,
                }
                for f in self.facts
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMemory":
        m = cls()
        m.schema_version = data.get("schema_version", 1)
        m.description = data.get("description", "")
        m.facts = [
            Fact(
                key=f["key"],
                value=f["value"],
                reason=f.get("reason", ""),
                updated_turn=f.get("updated_turn", 0),
            )
            for f in data.get("facts", [])
        ]
        return m

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectMemory":
        """Load from path. Returns an empty ProjectMemory if the file is missing or corrupt."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load project memory from %s: %s", path, exc)
            return cls()


# ── Meta helpers (persist display_name alongside the workspace dir) ────────────

def _read_meta(ws_root: Path) -> dict:
    path = ws_root / _META_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(ws_root: Path, display_name: str) -> None:
    path = ws_root / _META_FILENAME
    path.write_text(json.dumps({"display_name": display_name}, indent=2), encoding="utf-8")


def _display_name_for(ws_root: Path, slug: str) -> str:
    """Return persisted display name, or derive one from the slug if meta is missing."""
    meta = _read_meta(ws_root)
    return meta.get("display_name") or slug.replace("-", " ").title()


# ── Workspace ──────────────────────────────────────────────────────────────────

@dataclass
class Workspace:
    """A named, isolated project context."""

    name: str           # slug — directory name, e.g. "my-project"
    display_name: str   # human-readable label, e.g. "My Project"
    root: Path          # absolute path to workspace directory

    @property
    def files_dir(self) -> Path:
        """User-uploaded and generated files."""
        return self.root / "files"

    @property
    def knowledge_dir(self) -> Path:
        """ChromaDB RAG index for this workspace."""
        return self.root / "knowledge"

    @property
    def memory_path(self) -> Path:
        return self.root / "memory.json"

    def ensure_dirs(self) -> None:
        """Create all subdirectories if they don't exist."""
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def load_memory(self) -> ProjectMemory:
        return ProjectMemory.load(self.memory_path)

    def save_memory(self, memory: ProjectMemory) -> None:
        memory.save(self.memory_path)


# ── WorkspaceManager ───────────────────────────────────────────────────────────

class WorkspaceManager:
    """Creates and looks up workspaces under a shared root directory."""

    DEFAULT_NAME = "default"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def ensure_default(self) -> Workspace:
        """Return the default workspace, creating it if it doesn't exist."""
        return self.create_workspace(self.DEFAULT_NAME, "Default")

    def create_workspace(self, name: str, display_name: str = "") -> Workspace:
        """Create a new workspace directory tree. Returns it (existing or new).

        If `name` is not a valid slug it is slugified automatically.
        `display_name` is persisted in meta.json on first creation.
        On subsequent calls for an existing workspace the persisted name wins —
        a new display_name argument is ignored to avoid silent renames.
        """
        if not is_valid_slug(name):
            name = slugify(name)
        ws_root = self._root / name
        ws_root.mkdir(parents=True, exist_ok=True)

        meta_path = ws_root / _META_FILENAME
        if not meta_path.exists():
            # First creation — write the display name (derive if not given)
            resolved_display = display_name or name.replace("-", " ").title()
            _write_meta(ws_root, resolved_display)
        else:
            # Already exists — use the persisted name regardless of argument
            resolved_display = _display_name_for(ws_root, name)

        ws = Workspace(name=name, display_name=resolved_display, root=ws_root)
        ws.ensure_dirs()
        logger.debug("Workspace ready: '%s' (%s) at %s", name, resolved_display, ws_root)
        return ws

    def get_workspace(self, name: str) -> Optional[Workspace]:
        """Return workspace by slug, or None if the directory doesn't exist."""
        ws_root = self._root / name
        if not ws_root.is_dir():
            return None
        return Workspace(
            name=name,
            display_name=_display_name_for(ws_root, name),
            root=ws_root,
        )

    def get_by_display_name(self, display_name: str) -> Optional[Workspace]:
        """Look up a workspace by its display name."""
        for ws in self.list_workspaces():
            if ws.display_name == display_name:
                return ws
        return None

    def list_workspaces(self) -> list[Workspace]:
        """Return all workspaces sorted by name, using persisted display names."""
        result = []
        for path in sorted(self._root.iterdir()):
            if path.is_dir() and is_valid_slug(path.name):
                result.append(
                    Workspace(
                        name=path.name,
                        display_name=_display_name_for(path, path.name),
                        root=path,
                    )
                )
        return result

    def display_names(self) -> list[str]:
        """Return display names for all workspaces (for UI Select widgets)."""
        return [ws.display_name for ws in self.list_workspaces()]
