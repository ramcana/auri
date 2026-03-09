"""
Retriever: query the vector store using an embedding model.
"""

from __future__ import annotations

import logging

from auri.rag.embedder import Embedder
from auri.rag.store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return top-k chunks relevant to query.

        Each result: {text: str, source: str, score: float}
        score is cosine similarity in [0, 1]; higher is better.
        """
        embedding = self._embedder.embed_one(query)
        hits = self._store.query(embedding, top_k=top_k)
        logger.debug("Retrieved %d chunks for query: %.60s", len(hits), query)
        return hits

    def is_empty(self) -> bool:
        return self._store.count() == 0

    def list_sources(self) -> list[str]:
        return self._store.list_sources()
