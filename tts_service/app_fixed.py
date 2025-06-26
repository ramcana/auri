import os
import logging
import json
import numpy as np
import base64
import io
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for different TTS engines
try:
    import torch
    import torchaudio
    import soundfile as sf
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some TTS features will be disabled.")

# TTS configuration
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge_tts")  # Options: edge_tts, gtts, pyttsx3
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en-US-AriaNeural")
PIPER_API_URL = os.getenv("PIPER_API_URL", "http://localhost:5000")

# Create FastAPI app
app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech service for Voice Bot",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    pitch: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    
class TTSResponse(BaseModel):
    audio_data: str  # Base64 encoded audio
    format: str = "wav"
    duration_ms: int = 0
    
class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    gender: Optional[str] = None

class VoicesResponse(BaseModel):
    voices: List[VoiceInfo]

# TTS Engine implementations
async def tts_edge(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> Tuple[bytes, int]:
    """Generate speech using Microsoft Edge TTS (requires edge-tts package)"""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice, rate=f"+{int((speed-1)*50)}%")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Run the TTS generation
        start_time = time.time()
        await communicate.save(temp_path)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Read the audio file
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        return audio_data, duration_ms
    except ImportError:
        logger.error("edge-tts package not installed. Please install with 'pip install edge-tts'")
        raise HTTPException(status_code=500, detail="TTS engine not available: edge-tts package missing")
    except Exception as e:
        logger.error(f"Edge TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

async def tts_gtts(text: str, voice: str = "en", speed: float = 1.0) -> Tuple[bytes, int]:
    """Generate speech using Google Text-to-Speech"""
    try:
        from gtts import gTTS
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Run the TTS generation
        start_time = time.time()
        tts = gTTS(text=text, lang=voice[:2], slow=False)
        tts.save(temp_path)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Read the audio file
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        return audio_data, duration_ms
    except ImportError:
        logger.error("gTTS package not installed. Please install with 'pip install gtts'")
        raise HTTPException(status_code=500, detail="TTS engine not available: gTTS package missing")
    except Exception as e:
        logger.error(f"Google TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Fallback TTS using pyttsx3 (offline TTS)
async def tts_pyttsx3(text: str, voice: str = None, speed: float = 1.0) -> Tuple[bytes, int]:
    """Generate speech using pyttsx3 (offline TTS engine)"""
    try:
        import pyttsx3
        import io
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Set properties
        engine.setProperty('rate', int(engine.getProperty('rate') * speed))
        
        # Set voice if specified
        if voice:
            voices = engine.getProperty('voices')
            for v in voices:
                if voice in v.id:
                    engine.setProperty('voice', v.id)
                    break
        
        # Run the TTS generation
        start_time = time.time()
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Read the audio file
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        return audio_data, duration_ms
    except ImportError:
        logger.error("pyttsx3 package not installed. Please install with 'pip install pyttsx3'")
        raise HTTPException(status_code=500, detail="TTS engine not available: pyttsx3 package missing")
    except Exception as e:
        logger.error(f"pyttsx3 TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "tts_engine": TTS_ENGINE,
        "torch_available": TORCH_AVAILABLE
    }

@app.get("/voices", response_model=VoicesResponse)
async def list_voices():
    """List available voices for the current TTS engine"""
    voices = []
    
    try:
        if TTS_ENGINE == "edge_tts":
            try:
                import edge_tts
                voice_list = await edge_tts.list_voices()
                for voice in voice_list:
                    voices.append(VoiceInfo(
                        id=voice["ShortName"],
                        name=voice["DisplayName"],
                        language=voice["Locale"],
                        gender=voice["Gender"]
                    ))
            except ImportError:
                logger.warning("edge-tts not installed, returning empty voice list")
                
        elif TTS_ENGINE == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voice_list = engine.getProperty('voices')
                for voice in voice_list:
                    # Extract language from voice ID if possible
                    lang = "en"
                    if "+" in voice.id:
                        lang = voice.id.split("+")[1][:2]
                    
                    voices.append(VoiceInfo(
                        id=voice.id,
                        name=voice.name,
                        language=lang,
                        gender=None
                    ))
                engine.stop()
            except ImportError:
                logger.warning("pyttsx3 not installed, returning empty voice list")
        
        # If no voices were found or TTS engine doesn't support voice listing
        if not voices:
            # Add default voice as fallback
            voices.append(VoiceInfo(
                id=DEFAULT_VOICE,
                name="Default Voice",
                language="en",
                gender=None
            ))
            
        return VoicesResponse(voices=voices)
    
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text"""
    try:
        voice = request.voice or DEFAULT_VOICE
        text = request.text.strip()
        speed = request.speed or 1.0
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Synthesizing speech: {text[:50]}{'...' if len(text) > 50 else ''}")
        logger.info(f"Using TTS engine: {TTS_ENGINE}, voice: {voice}, speed: {speed}")
        
        # Select TTS engine based on configuration
        audio_data = None
        duration_ms = 0
        audio_format = "wav"
        
        if TTS_ENGINE == "edge_tts":
            audio_data, duration_ms = await tts_edge(text, voice, speed)
            audio_format = "mp3"
        elif TTS_ENGINE == "gtts":
            audio_data, duration_ms = await tts_gtts(text, voice, speed)
            audio_format = "mp3"
        else:  # Fallback to pyttsx3
            audio_data, duration_ms = await tts_pyttsx3(text, voice, speed)
            
        # Convert to base64
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        return TTSResponse(
            audio_data=base64_audio,
            format=audio_format,
            duration_ms=duration_ms
        )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting TTS service with engine: {TTS_ENGINE}")
    
    # Check if required packages are installed
    if TTS_ENGINE == "edge_tts":
        try:
            import edge_tts
            logger.info("edge-tts package found")
        except ImportError:
            logger.warning("edge-tts package not found. Install with: pip install edge-tts")
    
    elif TTS_ENGINE == "gtts":
        try:
            from gtts import gTTS
            logger.info("gTTS package found")
        except ImportError:
            logger.warning("gTTS package not found. Install with: pip install gtts")
    
    # Always check for fallback TTS
    try:
        import pyttsx3
        logger.info("pyttsx3 package found (fallback TTS available)")
    except ImportError:
        logger.warning("pyttsx3 package not found. Install with: pip install pyttsx3")
        
    logger.info("TTS service started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)
