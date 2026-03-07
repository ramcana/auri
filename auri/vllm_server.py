"""
VLLMServer: manages the vLLM OpenAI-compatible API server subprocess.

State machine:
    STOPPED → STARTING → READY
                       ↓
                    STOPPING → STOPPED
                       ↓
                    FAILED

Key behaviours:
- Port collision detection: if the port is already in use, probes /v1/models to
  determine whether our vLLM is already running (reuse) or something else (error).
- LRU LoRA management: tracks loaded LoRAs, evicts LRU when max_loaded_loras exceeded.
  Pinned LoRAs are never evicted.
- Observability: logs per-run to logs/vllm_<timestamp>.log and writes
  logs/active_vllm.json while running.
- Model verification: after /health returns 200, confirms served-model-name appears
  in GET /v1/models before declaring state=READY.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import signal
import socket
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from openai import AsyncOpenAI

from auri.model_manager import LoRAConfig, ModelConfig
from auri.settings import AppSettings

logger = logging.getLogger(__name__)


class VLLMState(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class VLLMServer:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._state = VLLMState.STOPPED
        self._process: Optional[asyncio.subprocess.Process] = None
        self._current_model: Optional[str] = None
        # OrderedDict used as LRU: most-recently-used at end
        self._loaded_loras: OrderedDict[str, LoRAConfig] = OrderedDict()
        self._pinned_loras: set[str] = set()
        self._log_file: Optional[Path] = None
        self._log_handle = None
        self._lock = asyncio.Lock()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> VLLMState:
        return self._state

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def loaded_lora_names(self) -> list[str]:
        return list(self._loaded_loras.keys())

    @property
    def base_url(self) -> str:
        return f"http://{self._settings.vllm_host}:{self._settings.vllm_port}/v1"

    def is_ready(self) -> bool:
        return self._state == VLLMState.READY

    # ── Public lifecycle API ──────────────────────────────────────────────────

    async def ensure_model(
        self,
        model_config: ModelConfig,
        lora_configs: list[LoRAConfig],
    ) -> None:
        """Start or restart vLLM to serve model_config with lora_configs.

        No-op if already READY with the same model.
        New LoRAs requested while READY that aren't loaded trigger a restart.
        """
        async with self._lock:
            requested_lora_names = {l.name for l in lora_configs}
            loaded_names = set(self._loaded_loras.keys())

            same_model = self._current_model == model_config.name
            loras_satisfied = requested_lora_names.issubset(loaded_names)

            if self._state == VLLMState.READY and same_model and loras_satisfied:
                logger.debug("ensure_model: already serving '%s' with required LoRAs — no-op", model_config.name)
                return

            await self._restart_locked(model_config, lora_configs)

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    # ── Internal lifecycle ────────────────────────────────────────────────────

    async def _restart_locked(
        self,
        model_config: ModelConfig,
        lora_configs: list[LoRAConfig],
    ) -> None:
        """Must be called while holding self._lock."""
        if self._state not in (VLLMState.STOPPED, VLLMState.FAILED):
            await self._stop_locked()
        await self._start_locked(model_config, lora_configs)

    async def _start_locked(
        self,
        model_config: ModelConfig,
        lora_configs: list[LoRAConfig],
    ) -> None:
        """Must be called while holding self._lock."""
        host = self._settings.vllm_host
        port = self._settings.vllm_port

        # Port collision check
        if self._port_in_use(host, port):
            logger.info("Port %d already in use — checking if it's our vLLM server", port)
            reuse = await self._try_reuse_existing(model_config)
            if reuse:
                return
            raise RuntimeError(
                f"Port {port} is occupied by another process. "
                "Stop it or change VLLM_PORT in .env before starting Auri."
            )

        # Determine which LoRAs to actually load (LRU management)
        loras_to_load = self._resolve_loras_to_load(lora_configs, model_config)

        cmd = self._build_command(model_config, loras_to_load)
        logger.info("Starting vLLM: %s", " ".join(cmd))

        self._state = VLLMState.STARTING

        # Open per-run log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self._settings.logs_dir / f"vllm_{timestamp}.log"
        self._settings.logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(log_path, "w")  # noqa: WPS515 (kept open for subprocess lifetime)
        self._log_file = log_path
        logger.info("vLLM logs → %s", log_path)

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=self._log_handle,
                stderr=self._log_handle,
            )
        except FileNotFoundError as exc:
            self._state = VLLMState.FAILED
            self._close_log()
            raise RuntimeError(
                "Could not start vLLM. Is it installed? Run: pip install vllm"
            ) from exc

        # Write observability file
        self._write_active_json(model_config, loras_to_load, cmd)

        # Wait for server readiness
        try:
            await self._wait_for_health()
            await self._verify_model_served(model_config.name)
        except Exception:
            self._state = VLLMState.FAILED
            self._delete_active_json()
            raise

        # Update internal state
        self._current_model = model_config.name
        self._loaded_loras = OrderedDict(
            (l.name, l) for l in loras_to_load
        )
        self._pinned_loras = set(model_config.pinned_loras)
        self._state = VLLMState.READY
        logger.info("vLLM READY — model='%s', loras=%s", model_config.name, [l.name for l in loras_to_load])

    async def _stop_locked(self) -> None:
        """Must be called while holding self._lock."""
        if self._state == VLLMState.STOPPED:
            return

        self._state = VLLMState.STOPPING
        proc = self._process

        if proc and proc.returncode is None:
            logger.info("Stopping vLLM subprocess (PID %d)...", proc.pid)
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                await asyncio.wait_for(proc.wait(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("vLLM did not exit after SIGTERM — sending SIGKILL")
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

        self._process = None
        self._current_model = None
        self._loaded_loras.clear()
        self._pinned_loras.clear()
        self._close_log()
        self._delete_active_json()
        self._state = VLLMState.STOPPED
        logger.info("vLLM stopped")

    # ── Command construction ──────────────────────────────────────────────────

    def _build_command(
        self,
        model_config: ModelConfig,
        lora_configs: list[LoRAConfig],
    ) -> list[str]:
        s = self._settings
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_config.path.resolve()),
            "--served-model-name", model_config.name,
            "--host", s.vllm_host,
            "--port", str(s.vllm_port),
            "--gpu-memory-utilization", str(model_config.gpu_memory_utilization),
            "--max-model-len", str(model_config.max_model_len),
            "--dtype", model_config.dtype,
            "--tensor-parallel-size", str(model_config.tensor_parallel_size),
        ]

        if lora_configs:
            cmd += [
                "--enable-lora",
                "--max-loras", str(s.max_loaded_loras),
                "--lora-modules",
            ]
            # Legacy format: name=path (compatible with all vLLM 0.x versions)
            # NOTE: After running the vLLM spike, switch to JSON format if needed:
            #   '{"name": "...", "path": "...", "base_model_name": "..."}'
            for lora in lora_configs:
                cmd.append(f"{lora.name}={lora.path.resolve()}")

        cmd.extend(model_config.extra_vllm_args)
        return cmd

    # ── LRU LoRA management ───────────────────────────────────────────────────

    def _resolve_loras_to_load(
        self,
        requested: list[LoRAConfig],
        model_config: ModelConfig,
    ) -> list[LoRAConfig]:
        """Determine which LoRAs to load, respecting max_loaded_loras and LRU policy."""
        max_loras = self._settings.max_loaded_loras
        pinned_names = set(model_config.pinned_loras)
        requested_names = {l.name for l in requested}

        # Always include pinned + explicitly requested
        to_load_names: OrderedDict[str, LoRAConfig] = OrderedDict()
        for lora in requested:
            if lora.name in pinned_names or lora.name in requested_names:
                to_load_names[lora.name] = lora

        # Fill remaining slots from pinned (already discovered by model_manager)
        for lora in requested:
            if lora.name in pinned_names and lora.name not in to_load_names:
                to_load_names[lora.name] = lora

        # Trim to max_loras (non-pinned evicted first)
        if len(to_load_names) > max_loras:
            non_pinned = [n for n in to_load_names if n not in pinned_names]
            while len(to_load_names) > max_loras and non_pinned:
                evict = non_pinned.pop(0)
                del to_load_names[evict]
                logger.info("LRU evicting LoRA '%s' (max_loaded_loras=%d)", evict, max_loras)

        return list(to_load_names.values())

    # ── Health / readiness checks ─────────────────────────────────────────────

    async def _wait_for_health(self) -> None:
        """Poll GET /health every 2 seconds until 200 or timeout."""
        health_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}/health"
        timeout_sec = self._settings.vllm_startup_timeout
        deadline = time.monotonic() + timeout_sec
        interval = 2.0

        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(connector=connector) as session:
            while time.monotonic() < deadline:
                # Check if subprocess exited unexpectedly
                if self._process and self._process.returncode is not None:
                    raise RuntimeError(
                        f"vLLM process exited with code {self._process.returncode}. "
                        f"Check {self._log_file} for details."
                    )
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            logger.debug("vLLM /health returned 200")
                            return
                except Exception:
                    pass
                await asyncio.sleep(interval)

        raise TimeoutError(
            f"vLLM did not become healthy within {timeout_sec}s. "
            f"Check {self._log_file} for details."
        )

    async def _verify_model_served(self, expected_name: str) -> None:
        """Confirm expected model name is in GET /v1/models response."""
        models_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}/v1/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"GET /v1/models returned HTTP {resp.status}")
                    data = await resp.json()
                    model_ids = [m.get("id", "") for m in data.get("data", [])]
                    logger.debug("vLLM served model IDs: %s", model_ids)
                    if expected_name not in model_ids:
                        raise RuntimeError(
                            f"vLLM is running but served model '{expected_name}' not found in /v1/models. "
                            f"Found: {model_ids}. Check --served-model-name in vLLM command."
                        )
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Could not reach vLLM /v1/models: {exc}") from exc

    # ── Port utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0

    async def _try_reuse_existing(self, model_config: ModelConfig) -> bool:
        """Return True if the existing server is already serving our model (reuse it)."""
        models_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}/v1/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    model_ids = [m.get("id", "") for m in data.get("data", [])]
                    if model_config.name in model_ids:
                        logger.info(
                            "Reusing existing vLLM server already serving '%s'",
                            model_config.name,
                        )
                        self._current_model = model_config.name
                        self._state = VLLMState.READY
                        return True
        except Exception:
            pass
        return False

    # ── OpenAI client ─────────────────────────────────────────────────────────

    def get_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=self.base_url, api_key="vllm")

    # ── Observability helpers ─────────────────────────────────────────────────

    def _write_active_json(
        self,
        model_config: ModelConfig,
        loras: list[LoRAConfig],
        cmd: list[str],
    ) -> None:
        active_path = self._settings.logs_dir / "active_vllm.json"
        payload = {
            "model_name": model_config.name,
            "model_path": str(model_config.path),
            "loaded_loras": [l.name for l in loras],
            "command": cmd,
            "pid": self._process.pid if self._process else None,
            "started_at": datetime.now().isoformat(),
            "log_file": str(self._log_file),
        }
        active_path.write_text(json.dumps(payload, indent=2))

    def _delete_active_json(self) -> None:
        active_path = self._settings.logs_dir / "active_vllm.json"
        try:
            active_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _close_log(self) -> None:
        if self._log_handle:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
