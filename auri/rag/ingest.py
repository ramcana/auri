"""
Ingestor: reads files, chunks them, embeds, and upserts into the vector store.

Supported file types: plain text formats + PDF (via pypdf).
"""

from __future__ import annotations

import logging
from pathlib import Path

from auri.rag.chunker import chunk_text
from auri.rag.embedder import Embedder
from auri.rag.store import VectorStore

logger = logging.getLogger(__name__)

# File extensions treated as plain text
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({
    ".txt", ".md", ".rst",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".env",
    ".sh", ".bash", ".zsh",
    ".html", ".css", ".sql",
    ".csv",
    ".pdf",  # extracted via pypdf
})

# Classification buckets — used for feedback and future chunking strategy
_FILE_TYPES: dict[str, str] = {
    ".py": "code", ".js": "code", ".ts": "code", ".jsx": "code", ".tsx": "code",
    ".sh": "code", ".bash": "code", ".zsh": "code", ".sql": "code",
    ".html": "code", ".css": "code",
    ".md": "docs", ".rst": "docs", ".txt": "docs", ".pdf": "docs",
    ".json": "config", ".yaml": "config", ".yml": "config",
    ".toml": "config", ".ini": "config", ".env": "config",
    ".csv": "data",
}


def _extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF using pypdf. Raises ImportError if not installed."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF ingestion. Install it with: pip install pypdf"
        )
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    if not pages:
        raise ValueError("No extractable text found in PDF (may be scanned/image-based).")
    return "\n\n".join(pages)


def classify_file_type(path: Path) -> str:
    """Return a broad category for a file: 'code', 'docs', 'config', 'data', or 'other'."""
    return _FILE_TYPES.get(path.suffix.lower(), "other")


class Ingestor:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        max_chunk_chars: int = 800,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._max_chunk_chars = max_chunk_chars

    def ingest_text(self, text: str, source: str) -> int:
        """
        Chunk, embed, and store raw text.

        source should be a human-readable identifier (file path, URL, etc.)
        Returns the number of chunks stored.
        """
        chunks = chunk_text(text, source, self._max_chunk_chars)
        if not chunks:
            logger.warning("No chunks produced from source: %s", source)
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed(texts)
        self._store.add(chunks, embeddings)
        logger.info("Ingested %d chunk(s) from '%s'", len(chunks), source)
        return len(chunks)

    def ingest_file(self, path: Path) -> int:
        """
        Read a file and ingest its contents.

        Raises ValueError for unsupported file types.
        Returns the number of chunks stored.
        """
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
        if suffix == ".pdf":
            text = _extract_pdf_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(text, source=str(path))

    def ingest_upload(self, temp_path: Path, original_name: str) -> int:
        """
        Ingest a Chainlit-uploaded file whose temp path may lack the proper extension.

        Uses original_name's suffix for type detection and as the source label.
        This handles Chainlit storing uploads as UUID blobs in .files/.

        Raises ValueError for unsupported file types.
        Returns the number of chunks stored.
        """
        suffix = Path(original_name).suffix.lower()
        if not suffix:
            raise ValueError(f"Cannot determine file type from name: '{original_name}'")
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
        if suffix == ".pdf":
            text = _extract_pdf_text(temp_path)
        else:
            text = temp_path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(text, source=original_name)

    @property
    def supported_suffixes(self) -> frozenset[str]:
        return SUPPORTED_SUFFIXES
