# TTS Service for Voice Bot

This service provides text-to-speech capabilities for the Voice Bot application, converting text responses from the LLM into spoken audio.

## Features

- Text-to-speech synthesis using Piper TTS (local TTS engine)
- Multiple voice options
- Base64-encoded audio response
- Health check endpoint
- Voice listing endpoint

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Piper TTS:
```bash
# Install Piper TTS from https://github.com/rhasspy/piper
# Download voice models as needed
```

3. Set up environment variables (optional):
```
TTS_ENGINE=piper
PIPER_API_URL=http://localhost:5000
DEFAULT_VOICE=en_US-lessac-medium
```

## API Endpoints

### Health Check
```
GET /health
```
Returns the status of the TTS service and the current engine.

### List Voices
```
GET /voices
```
Returns a list of available voices.

### Synthesize Speech
```
POST /synthesize
```

Request body:
```json
{
  "text": "Hello, how can I help you today?",
  "voice": "en_US-lessac-medium"  // Optional, uses DEFAULT_VOICE if not specified
}
```

Response:
```json
{
  "audio_data": "base64_encoded_audio_data",
  "format": "wav"
}
```

## Integration with Voice Bot

This service is designed to work with the Voice Bot application:
1. The main backend receives text responses from the LLM service
2. The backend sends the text to this TTS service
3. The TTS service returns base64-encoded audio
4. The backend forwards the audio to the frontend
5. The frontend plays the audio through the user's speakers
