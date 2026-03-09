"""
RetrievalTool: exposes the RAG knowledge base as a standard tool.

The model calls this tool when it needs to look up information from ingested
documents. Returns top-3 chunks above the similarity threshold with citations.
Results below MIN_SCORE are suppressed — weak context is worse than no context.
"""

from __future__ import annotations

import logging

from auri.rag.retriever import Retriever
from auri.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Cosine similarity floor. Chunks below this score are not passed to the model.
# all-MiniLM-L6-v2 range: ~0.0 (unrelated) → 1.0 (identical).
# 0.35 is conservative — only clear semantic matches pass.
_MIN_SCORE = 0.35


class RetrievalTool(BaseTool):
    name = "retrieve_knowledge"
    description = (
        "Search the local knowledge base for relevant information. "
        "Use this when the user references documents they have shared, "
        "or when a question may be answered by ingested files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant knowledge.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(self, query: str) -> ToolResult:  # type: ignore[override]
        if self._retriever.is_empty():
            return ToolResult(
                success=True,
                output={
                    "results": [],
                    "message": "Knowledge base is empty. Share a document to populate it.",
                },
                metadata={"retrieval_event": {
                    "query": query,
                    "chunks_returned": 0,
                    "sources": [],
                    "top_score": 0.0,
                    "below_threshold": False,
                }},
            )

        hits = self._retriever.retrieve(query, top_k=3)
        top_score = hits[0]["score"] if hits else 0.0

        # Threshold filter: discard weak matches
        strong = [h for h in hits if h["score"] >= _MIN_SCORE]

        if not strong:
            logger.debug(
                "Retrieval below threshold for query '%.60s' (best=%.2f, min=%.2f)",
                query, top_score, _MIN_SCORE,
            )
            return ToolResult(
                success=True,
                output={
                    "results": [],
                    "message": (
                        f"No strongly relevant information found (best match score: {top_score:.2f}). "
                        "Answer from your own knowledge if possible."
                    ),
                },
                metadata={"retrieval_event": {
                    "query": query,
                    "chunks_returned": 0,
                    "sources": [],
                    "top_score": top_score,
                    "below_threshold": True,
                }},
            )

        # Deduplicate sources preserving order of first appearance
        sources = list(dict.fromkeys(h["source"] for h in strong))
        results = [
            {"snippet": h["text"], "source": h["source"], "relevance": h["score"]}
            for h in strong
        ]
        return ToolResult(
            success=True,
            output={"results": results},
            metadata={"retrieval_event": {
                "query": query,
                "chunks_returned": len(strong),
                "sources": sources,
                "top_score": top_score,
                "below_threshold": False,
            }},
        )
