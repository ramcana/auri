"""
OllamaClient: thin wrapper around an AsyncOpenAI client pointed at Ollama's
OpenAI-compatible endpoint (http://localhost:11434/v1).
"""

from __future__ import annotations

import logging

import aiohttp
from openai import AsyncOpenAI

from auri.settings import AppSettings

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        # Ollama exposes an OpenAI-compatible API; the api_key is a required
        # placeholder by the openai library but is not validated by Ollama.
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        )

    @property
    def client(self) -> AsyncOpenAI:
        """Underlying AsyncOpenAI client for use by ModelRouter."""
        return self._client

    async def check_available(self) -> bool:
        """Probe Ollama's native root endpoint to confirm the daemon is running."""
        # Derive native URL from the OpenAI base_url
        # e.g. "http://localhost:11434/v1" → "http://localhost:11434/"
        base = self._settings.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            native_url = base[: -len("/v1")] + "/"
        else:
            native_url = base + "/"

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(native_url) as resp:
                    available = resp.status == 200
                    if available:
                        logger.debug("Ollama is available at %s", native_url)
                    else:
                        logger.warning("Ollama returned HTTP %d at %s", resp.status, native_url)
                    return available
        except Exception as exc:
            logger.warning("Ollama not reachable at %s: %s", native_url, exc)
            return False
