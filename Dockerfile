FROM python:3.11-slim

WORKDIR /app

# Build deps for chromadb / sentence-transformers (C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached independently of source changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source — configs are baked in but can be overridden by a bind mount
COPY auri/ ./auri/
COPY app.py .
COPY configs/ ./configs/

# Volume mount points (created empty; populated by docker-compose mounts)
RUN mkdir -p models/vllm models/ollama loras workspaces data/rag logs

EXPOSE 8000

# --host 0.0.0.0 is required — the default 127.0.0.1 is unreachable from outside the container
CMD ["chainlit", "run", "--host", "0.0.0.0", "app.py"]
