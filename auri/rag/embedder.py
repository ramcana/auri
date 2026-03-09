"""
Embedding model wrapper using sentence-transformers.

all-MiniLM-L6-v2 (~90 MB) is the default: fast, good retrieval quality,
no API key required. Downloaded automatically on first use to ~/.cache/.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        # Import deferred so the module loads even if sentence-transformers
        # is not installed — the error surfaces only when Embedder is instantiated.
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of float vectors."""
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()
