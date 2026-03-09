"""
ChromaDB vector store wrapper.

Persistent collection stored in data/rag/ under the project root.
Uses cosine similarity (distance → similarity conversion on retrieval).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "auri_knowledge"


class VectorStore:
    def __init__(self, persist_dir: Path) -> None:
        import chromadb  # type: ignore

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            _COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore ready — %d chunks indexed", self._col.count())

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Upsert chunks into the collection. Re-ingesting the same source is safe."""
        if not chunks:
            return
        ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"source": c["source"], "chunk_index": str(c["chunk_index"])}
            for c in chunks
        ]
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """Return top-k chunks by cosine similarity. Empty list if store is empty."""
        count = self._col.count()
        if count == 0:
            return []
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta["source"],
                # ChromaDB cosine distance: 0 = identical, 2 = opposite.
                # Convert to similarity in [0, 1].
                "score": round(1.0 - dist / 2.0, 4),
            })
        return hits

    def list_sources(self) -> list[str]:
        """Return sorted list of distinct source paths in the index."""
        if self._col.count() == 0:
            return []
        result = self._col.get(include=["metadatas"])
        return sorted({m["source"] for m in result["metadatas"]})

    def count(self) -> int:
        return self._col.count()
