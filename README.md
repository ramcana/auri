# Auri

**Auri** is your local AI assistant powered by [vLLM](https://github.com/vllm-project/vllm) and [Ollama](https://ollama.com). It provides a Chainlit-based chat interface to interact with large language models locally, supporting both GPU-accelerated models via vLLM and lightweight models via Ollama.

## Features

- **Dual Backend Support**: Run large models on GPU with vLLM or smaller models with Ollama.
- **LoRA Adapters**: Customize model behavior with Low-Rank Adaptations.
- **Easy Model Management**: Add models by placing them in the appropriate directories.
- **Web Interface**: Built with Chainlit for a seamless chat experience.

## Prerequisites

- Python 3.8+
- [vLLM](https://github.com/vllm-project/vllm) (install with CUDA support for GPU acceleration)
- [Ollama](https://ollama.com/download) (standalone binary)

## Installation

1. Clone or download this repository.

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install vLLM separately (ensure CUDA is available for GPU support):
   ```bash
   pip install vllm
   ```

4. Install and start Ollama:
   - Download from [ollama.com](https://ollama.com/download)
   - Follow the installation instructions for your OS.

## Usage

1. Start the application:
   ```bash
   chainlit run app.py
   ```

2. Open your browser to the provided URL (usually `http://localhost:8000`).

3. Click the **settings icon** (bottom-left of the chat bar) to open the configuration panel.

4. Select your **Model** — large models run on vLLM (GPU-accelerated), small ones via Ollama.

5. Optionally pick a **LoRA Adapter** to specialize the model's behavior.

6. Start chatting!

## Adding Models

### vLLM (GPU models)
Place a HuggingFace model directory inside `models/vllm/`:
```
models/vllm/my-model/
    config.json
    *.safetensors
```

### Ollama models
Create a directory inside `models/ollama/`. Optionally add a `model.txt` with the Ollama tag if the directory name differs from the tag (e.g., `phi3:mini`):
```
models/ollama/phi3-mini/
    model.txt   ← contains "phi3:mini"
```

## Adding LoRA Adapters

Place a HuggingFace PEFT adapter directory inside `loras/`:
```
loras/my-lora/
    adapter_config.json
    adapter_model.safetensors
```

Then add it to `configs/models.yaml` under the model's `compatible_loras` list.

## Configuration

- Models and LoRAs are configured in `configs/models.yaml`.
- Environment variables can be set in `.env` (copy from `.env.example`).

## Tips

- Switching models may take a moment as the vLLM server restarts — a status message will appear.
- vLLM models marked **[offline]** indicate the Ollama daemon isn't running.
- Check `logs/vllm_*.log` and `logs/active_vllm.json` to inspect the running vLLM configuration.
- Chainlit is single-process by default; for production, consider scaling options.

## License

[Add license if applicable]

## Contributing

[Add contribution guidelines if applicable]