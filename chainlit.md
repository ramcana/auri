# Welcome to Auri

**Auri** is your local AI assistant powered by [vLLM](https://github.com/vllm-project/vllm) and [Ollama](https://ollama.com).

## Getting Started

1. Click the **settings icon** (bottom-left of the chat bar) to open the configuration panel.
2. Select your **Model** — large models run on vLLM (GPU-accelerated), small ones via Ollama.
3. Optionally pick a **LoRA Adapter** to specialise the model's behaviour.
4. Start chatting!

## Adding Models

**vLLM (GPU models):** Place a HuggingFace model directory inside `models/vllm/`:
```
models/vllm/my-model/
    config.json
    *.safetensors
```

**Ollama models:** Create a directory inside `models/ollama/`. Optionally add a `model.txt` with the Ollama tag if the directory name differs from the tag (e.g. `phi3:mini`):
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

## Tips

- Switching models may take a moment as the vLLM server restarts — a status message will appear.
- vLLM models marked **[offline]** indicate the Ollama daemon isn't running.
- Check `logs/vllm_*.log` and `logs/active_vllm.json` to inspect the running vLLM configuration.
- Start Auri with: `chainlit run app.py`
