# Auri - Modular Context-Aware Voice Assistant

Auri is a modular, real-time voice assistant with a modern Electron/React desktop UI and a robust Python FastAPI backend. It features a pluggable tool system, persistent conversational memory, and user profile management for highly personalized, context-rich interactions.

---

## 🚀 Features

- **Real-time Voice and Text Chat:** Natural conversation with speech-to-text (STT) and text-to-speech (TTS) microservices.
- **Modular Tool System:** Easily extendable tool modules (e.g., news, file conversion, Wikipedia, world time) that fetch live data and augment LLM responses.
- **Contextual Memory:** Persistent user memory and conversation facts, summarized and injected into LLM prompts for better multi-turn context retention.
- **User Profiles & Logging:** Per-user profile preferences and interaction logs, enabling adaptive and personalized responses.
- **Modern UI:** Electron/React frontend with custom branding, logo, tagline, and accessible controls.
- **Environment Variable Management:** Secure, scalable configuration via `.env` and `python-dotenv`.

---

## 🏗️ Architecture Overview

- **Backend:**
  - FastAPI microservices for LLM, tool routing, user memory/profile, and logging.
  - Modular backend: add/remove tool modules in `backend/modules/`.
  - JSON-based persistent storage for user profiles, logs, and facts (DB migration ready).
- **Frontend:**
  - Electron/React app (`renderer/`) with modern chat UI, theme toggle, and branded header.
  - Real-time communication via WebSocket.
- **Microservices:**
  - `tts_service/`, `stt_service/`: Standalone Python microservices for TTS/STT.

---

## 📦 Project Structure

- `backend/` — FastAPI backend, tool modules, memory, logging
- `renderer/` — React UI (main app, components, styling)
- `electron/` — Electron main process and assets (logo, etc.)
- `tts_service/`, `stt_service/` — Microservices for TTS/STT
- `.env`, `.env.example` — Environment variables
- `requirements.txt` — Python dependencies
- `package.json` — Node.js/React dependencies

---

## ⚙️ Setup & Usage

### Prerequisites
- Node.js 18+
- Python 3.9+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/ramcana/auri.git
cd auri
```

### 2. Install Dependencies
```bash
# Node.js packages
npm install

# Python backend
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and update API keys and settings as needed.

### 4. Start the App
```bash
npm run dev
```
This launches the Electron app, backend, and frontend with hot reload.

---

## 📝 Environment Variables
- All sensitive keys and config are managed in `.env` (see `.env.example` for template)
- Example: `NEWS_API_KEY`, service ports, etc.

---

## 🛠️ Extending & Customizing
- **Add new tools:** Place new modules in `backend/modules/` and register them in the backend.
- **Customize UI:** Edit React components in `renderer/src/` and assets in `electron/img/`.
- **User memory:** All user profile and fact storage is JSON-based for easy migration to a database.

---

## 🧹 Housekeeping
- **Unused models:** The legacy `models/dia/` directory is not used and should be deleted or gitignored.
- **Logs, facts, and user data** are not tracked in git.

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit and push your changes
4. Open a Pull Request

---

## 📄 License
MIT License — see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the Auri Team**
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Install Whisper Models
```bash
# Download Whisper models
./scripts/install-models.sh
```

### 4. Start Backend Services
```bash
# Start all services
docker-compose up -d

# Or start individual services
docker-compose up audio-service llm-service data-service config-service
```

### 5. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 6. Start Frontend Application
```bash
# Development mode
npm run dev

# Production build
npm run build
npm run start
```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=voicebot
POSTGRES_USER=voicebot
POSTGRES_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# LLM Providers
OLLAMA_BASE_URL=http://localhost:11434

# Audio Service
WHISPER_MODEL=base
AUDIO_SAMPLE_RATE=16000

# Services
AUDIO_SERVICE_PORT=8001
LLM_SERVICE_PORT=8002
DATA_SERVICE_PORT=8003
CONFIG_SERVICE_PORT=8004
```

### LLM Provider Configuration
```json
{
  "default_provider": "ollama",
  "providers": {
    "ollama": {
      "base_url": "http://localhost:11434",
      "model": "llama2",
      "timeout": 30
    }
  }
}
```

## 🚀 Usage

### Basic Voice Interaction
1. Launch the desktop application
2. Click the microphone button to start recording
3. Speak your question or command
4. The AI will respond with both text and synthesized speech

### Switching LLM Providers
1. Open Settings in the application
2. Select your preferred LLM provider
3. Configure provider-specific settings
4. Save and restart the conversation

### API Usage
```bash
# Test Audio Service
curl -X POST "http://localhost:8001/stt" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav"

# Test LLM Service
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "provider": "ollama"}'
```

## 🛠️ Development

### Running Individual Services
```bash
# Audio Service
cd services/audio-service
python -m uvicorn main:app --reload --port 8001

# LLM Service
cd services/llm-service
python -m uvicorn main:app --reload --port 8002

# Data Service
cd services/data-service
python -m uvicorn main:app --reload --port 8003
```

### Ollama LLM Integration

The application integrates with Ollama to provide local LLM capabilities:

```bash
# Install Ollama from https://ollama.ai

# Pull models (examples)
ollama pull llama3.1:latest
ollama pull mistral:7b-instruct
ollama pull tinyllama:latest

# Start Ollama service
ollama serve
```

The LLM service connects to Ollama on port 11434 by default and can be configured via environment variables:

```env
OLLAMA_API_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.1:latest
```

### Frontend Development
```bash
cd frontend
npm run dev  # Hot reload enabled
```

### Adding New LLM Providers
1. Create new provider class in `services/llm-service/app/providers/`
2. Implement the `BaseLLMProvider` interface
3. Register provider in `provider_factory.py`
4. Update configuration schema

## 📚 API Documentation

### Audio Service Endpoints
- `POST /stt` - Speech to text conversion
- `POST /tts` - Text to speech conversion
- `GET /health` - Service health check

### LLM Service Endpoints
- `POST /chat` - Send message to LLM
- `GET /providers` - List available providers
- `POST /providers/{provider}/config` - Update provider config

### Data Service Endpoints
- `GET /conversations` - List conversations
- `POST /conversations` - Create new conversation
- `GET /conversations/{id}` - Get conversation details

## 🔒 Security

- API keys stored in environment variables
- Input validation on all endpoints
- Rate limiting implemented
- CORS properly configured
- Secure audio file handling

## 🐛 Troubleshooting

### Common Issues

**Microphone not working:**
- Check browser permissions
- Ensure HTTPS in production
- Verify audio device selection

**Ollama connection failed:**
- Ensure Ollama is running: `ollama serve`
- Check firewall settings
- Verify model is installed: `ollama list`

**Whisper model not found:**
- Run model installation script
- Check model path in configuration
- Verify disk space availability

**Docker services not starting:**
- Check Docker daemon is running
- Verify port availability
- Review docker-compose logs

## 📈 Performance Optimization

- Use smaller Whisper models for faster inference
- Implement audio chunking for long recordings
- Cache frequent LLM responses
- Use connection pooling for database
- Enable Redis caching for configurations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Create an issue for bug reports
- Join our Discord community
- Check the [Wiki](https://github.com/yourusername/voice-bot/wiki) for detailed guides
- Email: support@voicebot.example.com

## 🗺️ Roadmap

- [ ] Mobile app support (React Native)
- [ ] Voice cloning capabilities
- [ ] Multi-language support
- [ ] Plugin system for custom integrations
- [ ] Cloud deployment templates
- [ ] Advanced conversation analytics

---

**Built with ❤️ by the Voice Bot Team**