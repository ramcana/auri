import os
import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import the WebSocket handler
from websocket_handler import websocket_endpoint, active_connections

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://localhost:8001/transcribe/")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:11434/api/generate")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8003/synthesize")

# Create FastAPI app
app = FastAPI(
    title="Voice Bot Backend",
    description="Handles WebSocket connections for voice and text input processing",
    version="0.1.0"
)

# Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Voice Bot Backend API"}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "services": {
            "llm": LLM_SERVICE_URL,
            "tts": TTS_SERVICE_URL,
            "stt": STT_SERVICE_URL
        }
    }

# CORS configuration - Allow all origins for WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the WebSocket endpoint
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for client connections"""
    await websocket_endpoint(websocket)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
