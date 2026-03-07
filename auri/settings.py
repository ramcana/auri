"""
App-wide settings loaded from environment variables and .env file.
All paths resolved to absolute at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels up from this file (Auri/auri/settings.py -> Auri/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class AppSettings:
    # --- Paths ---
    project_root: Path
    models_vllm_dir: Path
    models_ollama_dir: Path
    loras_dir: Path
    config_path: Path
    logs_dir: Path

    # --- vLLM server ---
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_startup_timeout: int = 180  # seconds to wait for server ready

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"  # placeholder required by openai client

    # --- LoRA management ---
    max_loaded_loras: int = 4  # LRU cap for simultaneously loaded LoRAs

    # --- Misc ---
    log_level: str = "INFO"


def load_settings() -> AppSettings:
    """Load settings from environment variables (via python-dotenv) and return AppSettings."""
    root = PROJECT_ROOT
    return AppSettings(
        project_root=root,
        models_vllm_dir=root / "models" / "vllm",
        models_ollama_dir=root / "models" / "ollama",
        loras_dir=root / "loras",
        config_path=root / "configs" / "models.yaml",
        logs_dir=root / "logs",
        vllm_host=os.getenv("VLLM_HOST", "127.0.0.1"),
        vllm_port=int(os.getenv("VLLM_PORT", "8000")),
        vllm_startup_timeout=int(os.getenv("VLLM_STARTUP_TIMEOUT", "180")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        max_loaded_loras=int(os.getenv("MAX_LOADED_LORAS", "4")),
        log_level=os.getenv("AURI_LOG_LEVEL", "INFO"),
    )
