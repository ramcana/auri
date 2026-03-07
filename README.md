# Auri

A local AI chat assistant with a [Chainlit](https://chainlit.io) interface, dual inference backends, and ComfyUI-style model management — drop models and LoRAs into folders and they appear automatically.

| Backend | Best for | Discovery |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Large models, GPU-accelerated, LoRA adapters | `models/vllm/<name>/` |
| [Ollama](https://ollama.com) | Small/quantized models, CPU-friendly | Auto-detected from `ollama list` |

---

## Features

- **Dual inference backends** — vLLM for GPU-accelerated large models, Ollama for small/quantized models, routed automatically
- **Dynamic LoRA adapters** — drop PEFT adapter folders into `loras/` and assign them to models in `configs/models.yaml`
- **Auto-discovery** — Ollama models appear on startup without any config; new `ollama pull` models show up after a restart
- **Sidebar controls** — Model, LoRA, Temperature, and Max Tokens selectable per conversation
- **Per-conversation history** — each chat session maintains its own message history and system prompt
- **Observability** — per-run vLLM logs in `logs/` and a live `logs/active_vllm.json` showing the running config

---

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (for vLLM models)
- [Ollama](https://ollama.com/download) daemon running (for Ollama models)
- [vLLM](https://github.com/vllm-project/vllm) installed separately (for GPU models)

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Auri dependencies
pip install -r requirements.txt

# 3. Install vLLM (only needed if you plan to use GPU models)
pip install vllm

# 4. Copy environment config and edit if needed
cp .env.example .env
```

---

## Running

```bash
chainlit run -h app.py
```

Then open `http://localhost:8000` in your browser.

> **WSL users:** use `-h` (headless) to prevent Chainlit from trying to auto-open a browser via Windows interop.

---

## Adding Models

### Ollama models

Just pull them — they appear automatically on the next restart:

```bash
ollama pull llama3.2
```

To customise the display name or system prompt, add an entry to `configs/models.yaml`:

```yaml
models:
  llama3-2:
    backend: ollama
    ollama_model_name: "llama3.2:latest"
    display_name: "LLaMA 3.2"
    system_prompt: "You are Auri, a helpful assistant."
```

### vLLM models (GPU)

Place a HuggingFace model directory in `models/vllm/` — it needs a `config.json` or at least one `.safetensors` file:

```
models/vllm/mistral-7b/
    config.json
    model.safetensors
```

Then add a YAML entry:

```yaml
models:
  mistral-7b:
    backend: vllm
    path: "models/vllm/mistral-7b"
    display_name: "Mistral 7B"
    gpu_memory_utilization: 0.85
    max_model_len: 8192
    dtype: "bfloat16"
```

---

## Adding LoRA Adapters

Drop a HuggingFace PEFT adapter directory into `loras/` — it must contain `adapter_config.json`:

```
loras/my-lora/
    adapter_config.json
    adapter_model.safetensors
```

Then reference it from the base model's entry in `configs/models.yaml`:

```yaml
models:
  mistral-7b:
    backend: vllm
    compatible_loras: [my-lora]
    pinned_loras: [my-lora]      # load at vLLM startup (optional)
```

---

## Configuration

All configuration lives in two places:

| File | Purpose |
|---|---|
| `configs/models.yaml` | Model metadata, LoRA compatibility, system prompts, vLLM tuning |
| `.env` | Runtime settings — vLLM host/port, Ollama URL, log level |

### Key `models.yaml` fields

```yaml
schema_version: 1

defaults:
  max_tokens: 2048
  temperature: 0.7
  max_loaded_loras: 4      # LRU cap for simultaneously loaded LoRAs

models:
  my-model:
    backend: vllm | ollama
    display_name: "Human-readable name"
    system_prompt: "Custom system prompt for this model."
    compatible_loras: [lora-name-1, lora-name-2]
    pinned_loras: [lora-name-1]    # always loaded at vLLM startup
    # vLLM-specific:
    gpu_memory_utilization: 0.85
    max_model_len: 8192
    dtype: bfloat16
    tensor_parallel_size: 1
    extra_vllm_args: []
    # Ollama-specific:
    ollama_model_name: "tag:variant"
```

### Key `.env` settings

```bash
VLLM_HOST=127.0.0.1
VLLM_PORT=8000
VLLM_STARTUP_TIMEOUT=180
OLLAMA_BASE_URL=http://localhost:11434/v1
MAX_LOADED_LORAS=4
AURI_LOG_LEVEL=INFO
```

---

## Project Structure

```
Auri/
├── app.py                  # Chainlit entry point
├── chainlit.md             # In-app welcome screen
├── configs/
│   └── models.yaml         # Model and LoRA configuration
├── models/
│   ├── vllm/               # HuggingFace model directories
│   └── ollama/             # Optional: override dirs with model.txt
├── loras/                  # PEFT LoRA adapter directories
├── logs/                   # Per-run vLLM logs + active_vllm.json
└── auri/
    ├── settings.py         # AppSettings, .env loader
    ├── model_manager.py    # Filesystem + Ollama daemon discovery
    ├── vllm_server.py      # vLLM subprocess lifecycle & state machine
    ├── ollama_client.py    # Ollama OpenAI-compatible client
    └── router.py           # Request routing & streaming
```

---

## Debugging

- **vLLM won't start** — check `logs/vllm_<timestamp>.log` for the full output
- **Which model is loaded** — inspect `logs/active_vllm.json` for the current model, LoRAs, PID, and exact CLI args
- **Ollama models missing** — run `ollama list` to confirm they're pulled, then restart Auri
- **Port conflict** — set `VLLM_PORT` in `.env` to a free port; Chainlit defaults to 8000 for its own UI

> **LoRA routing note:** vLLM selects a LoRA by passing its name as the `model` field in the API request. Verify this works with your installed vLLM version before relying on it — see `auri/router.py:_resolve_vllm_model_name()` to adjust if needed.
