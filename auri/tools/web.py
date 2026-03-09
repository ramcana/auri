"""
Web search tool — uses the duckduckgo-search package for real web results.

No API key required. Runs the sync DDGS client in a thread pool so it
doesn't block the event loop. Returns structured JSON (titles, URLs, snippets).
"""

from __future__ import annotations

import asyncio
import logging

from auri.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current information such as news, facts, prices, or live data. "
        "Use this when the answer requires up-to-date information not in your training data. "
        "Returns titles, URLs, and text snippets for the top results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str, max_results: int = 5) -> ToolResult:  # type: ignore[override]
        max_results = min(max(1, max_results), 10)
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="duckduckgo-search is not installed. Run: pip install duckduckgo-search",
            )

        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            raw = await asyncio.get_event_loop().run_in_executor(None, _search)
        except Exception as exc:
            logger.warning("Web search failed for query '%s': %s", query[:60], exc)
            return ToolResult(success=False, output=None, error=f"Search failed: {exc}")

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
        logger.debug("Web search returned %d results for: %s", len(results), query[:60])
        return ToolResult(success=True, output={"query": query, "results": results})
