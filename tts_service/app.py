import os
import logging
import json
import numpy as np
import base64
import io
import tempfile
import time
import wave
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

# Environment variables
DIA_MODEL_PATH = os.getenv('DIA_MODEL_PATH', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'dia'))
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'en-US-JennyNeural')
TTS_SPEED = float(os.getenv('TTS_SPEED', '1.0'))
TTS_PITCH = float(os.getenv('TTS_PITCH', '0.0'))
# Force Edge TTS as default for better performance
os.environ['TTS_ENGINE'] = 'edge'  # Force Edge TTS regardless of environment variable
TTS_ENGINE = 'edge'  # Default to Edge TTS for better performance
logger.info(f"Setting TTS_ENGINE to: {TTS_ENGINE} (forced)")
# This ensures we always use Edge TTS for better performance

PIPER_API_URL = os.getenv("PIPER_API_URL", "http://localhost:5000")

# Since TTS_ENGINE is forced to 'edge', we only need to import edge_tts
# This significantly speeds up the service startup time.

EDGE_TTS_AVAILABLE = False
try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
    logger.info("Edge TTS is available and will be used as the primary engine.")
except ImportError:
    logger.error("FATAL: edge-tts package not found. The service cannot start without it.")
    # In a real production scenario, you might want to exit here.
    # For this context, we'll let it continue but it will fail on request.

# All other TTS engines are disabled to ensure fast startup
DIA_AVAILABLE = False
GTTS_AVAILABLE = False
PYTTSX3_AVAILABLE = False
PIPER_AVAILABLE = False
TORCH_AVAILABLE = False

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

def preprocess_text_for_tts(text: str) -> str:
    """Preprocess text to make it more suitable for TTS
    
    Args:
        text: Raw text to process
        
    Returns:
        Processed text ready for TTS
    """
    if not text:
        return ""
    
    # Replace problematic characters
    replacements = [
        ('&', ' and '),       # Replace ampersands
        ('...', '.'),         # Ellipsis becomes period
        ('--', ' - '),        # Double dash becomes spaced dash
        ('_', ' '),           # Underscores become spaces
        ('*', ''),            # Remove asterisks (markdown formatting)
        ('`', ''),            # Remove backticks (code formatting)
        ('|', ' '),           # Remove vertical bars (table formatting)
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c.isspace())
    
    return text

# Models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    pitch: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    engine: Optional[str] = None  # Optional engine override
    
class TTSResponse(BaseModel):
    audio_data: str  # Base64 encoded audio
    format: str = "wav"
    duration_ms: int = 0
    task_id: Optional[str] = None  # For background tasks
    status: Optional[str] = None  # For background tasks
    
class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    gender: Optional[str] = None

class VoicesResponse(BaseModel):
    voices: List[VoiceInfo]

# Global model instances for TTS engines
dia_model = None
dia_device = None  # Global device for Dia TTS model

# TTS Engine implementations

async def tts_edge(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, pitch: float = 0.0) -> Tuple[bytes, int, str]:
    """
    Generate speech using Edge TTS
    
    Args:
        text: Text to synthesize
        voice: Voice to use (e.g., 'en-US-JennyNeural')
        speed: Speed factor (1.0 is normal)
        pitch: Pitch adjustment (0.0 is normal)
        
    Returns:
        Tuple of (audio_bytes, duration_ms, format)
    """
    # Debug logging
    logger.info(f"tts_edge called with: text length={len(text)}, voice={voice}, speed={speed}, pitch={pitch}")

    if not EDGE_TTS_AVAILABLE:
        logger.warning("Edge TTS is not available. Falling back to pyttsx3.")
        return await tts_pyttsx3(text, voice, speed, pitch)

    try:
        # Limit text length to prevent TTS issues
        MAX_TTS_LENGTH = 1000  # Maximum characters for a single TTS request
        
        if len(text) > MAX_TTS_LENGTH:
            logger.warning(f"Text exceeds maximum length ({len(text)} > {MAX_TTS_LENGTH}). Truncating to prevent TTS issues.")
            # Find the last sentence boundary before the limit
            truncation_point = MAX_TTS_LENGTH
            for end_char in ['.', '!', '?', ';', ':', '\n']:
                last_boundary = text[:MAX_TTS_LENGTH].rfind(end_char)
                if last_boundary > 0:
                    truncation_point = last_boundary + 1
                    break
            text = text[:truncation_point]
            logger.info(f"Truncated text to {len(text)} characters at sentence boundary")
        
        # Normalize text to prevent gibberish
        # Replace problematic characters
        text = text.replace('...', '.')
        text = text.replace('--', ' - ')
        
        # Remove any non-printable characters that might cause issues
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        logger.info(f"Generating TTS with Edge TTS for text length: {len(text)} with speed {speed}")

        # Clamp speed and pitch to valid ranges for safety
        speed_val = max(0.5, min(2.0, speed))
        pitch_val = max(0.5, min(1.5, pitch)) # Pitch is more sensitive

        # Use plain text instead of SSML to avoid any issues with XML parsing
        # We'll control rate and pitch through the communicate options if available
        logger.info(f"Using plain text for TTS: {text[:50]}...")
        
        # Create communication object with plain text
        # For Edge TTS, we can set rate and pitch directly
        try:
            # Try the newer API with rate and pitch options
            communicate = edge_tts.Communicate(text, voice, rate=f"+{int((speed_val-1)*100)}%", pitch=f"+{int(pitch_val*10)}Hz")
            logger.info("Using newer Edge TTS API with direct rate and pitch control")
        except TypeError:
            # Fall back to the older API without rate and pitch
            communicate = edge_tts.Communicate(text, voice)
            logger.info("Using older Edge TTS API without rate and pitch control")

        # Start measuring time
        start_time = time.time()

        # Generate audio
        audio_data = bytearray()
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
        except Exception as stream_exc:
            logger.error(f"Exception during communicate.stream(): {stream_exc}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Calculate generation time
        gen_time = time.time() - start_time
        logger.info(f"Edge TTS generation completed in {gen_time:.2f}s")

        # MP3 audio: cannot use wave.open for duration, set duration_ms to 0
        audio_bytes = bytes(audio_data)
        logger.info(f"tts_edge: audio_bytes length before return: {len(audio_bytes)}")
        if len(audio_bytes) < 1000:
            logger.warning(f"tts_edge: Generated very short or empty audio! (len={len(audio_bytes)})")
        duration_ms = 0
        logger.warning("tts_edge: Audio is MP3, duration not calculated.")
        logger.info(f"Generated {len(audio_bytes)} bytes of audio, duration: {duration_ms}ms")
        audio_format = "mp3"
        return audio_bytes, duration_ms, audio_format

    except Exception as e:
        logger.error(f"Error in Edge TTS generation: {e}, falling back to pyttsx3")
        import traceback
        logger.error(traceback.format_exc())
        return await tts_pyttsx3(text, voice, speed, pitch)


async def tts_dia(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, pitch: float = 1.0) -> Tuple[bytes, int]:
    """
    Generate speech using Nari Labs' Dia TTS model
    
    Args:
        text: Text to synthesize
        voice: Not used for Dia (kept for API compatibility)
        speed: Speed factor (0.5 to 2.0)
        pitch: Not directly used by Dia (kept for API compatibility)
        
    Returns:
        Tuple of (audio_bytes, duration_ms)
    """
    global dia_model, dia_device
    
    if not DIA_AVAILABLE:
        logger.warning("Dia TTS is not available. Falling back to Edge TTS.")
        return await tts_edge(text, voice, speed)
    
    # Initialize model if not already loaded
    if dia_model is None:
        try:
            logger.info("Loading Dia TTS model...")
            model_load_start_time = time.time()
            
            # Force check for CUDA availability and log detailed information
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            
            if cuda_available:
                # Get CUDA device count and properties
                device_count = torch.cuda.device_count()
                logger.info(f"CUDA device count: {device_count}")
                
                # Get device properties for each GPU
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    logger.info(f"GPU {i}: {device_name}")
                
                # Use CUDA device 0
                dia_device = torch.device("cuda:0")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                # Log CUDA version
                if hasattr(torch.version, 'cuda'):
                    logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                dia_device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
                
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"Using device: {dia_device} for Dia TTS")
            
            # First try loading from local path
            try:
                if os.path.exists(dia_path):
                    logger.info(f"Attempting to load from local path: {dia_path}")
                    
                    # Load model with device specification
                    dia_model = Dia.from_pretrained(dia_path, device=dia_device)
                    
                    # Apply model optimization for inference
                    if torch.cuda.is_available():
                        logger.info("Optimizing model for GPU inference...")
                        try:
                            # Set model to evaluation mode for inference
                            dia_model.eval()
                            logger.info("Model set to evaluation mode")
                            
                            # Try to optimize with torch.compile if available (PyTorch 2.0+)
                            if hasattr(torch, 'compile'):
                                try:
                                    logger.info("Applying torch.compile optimization")
                                    dia_model = torch.compile(dia_model, mode="reduce-overhead")
                                    logger.info("Model compiled with torch.compile")
                                except Exception as compile_err:
                                    logger.warning(f"Could not compile model: {compile_err}")
                        except Exception as opt_err:
                            logger.warning(f"Optimization failed: {opt_err}, continuing with standard model")
                    
                    logger.info("Successfully loaded Dia model from local path")
                else:
                    logger.info(f"Local path {dia_path} not found, loading from Hugging Face")
                    dia_model = Dia.from_pretrained("Deci/dia", device=dia_device)
                    logger.info("Successfully loaded Dia model from Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to load from local path: {e}")
                logger.info("Attempting to load from Hugging Face")
                try:
                    dia_model = Dia.from_pretrained("Deci/dia", device=dia_device)
                    logger.info("Successfully loaded Dia model from Hugging Face")
                except Exception as e:
                    raise RuntimeError(f"Failed to load Dia model: {e}") from e
            
            load_time = time.time() - model_load_start_time
            logger.info(f"Dia model loaded in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load Dia model: {e}")
            logger.error("Falling back to Edge TTS")
            return await tts_edge(text, voice, speed)
    
    # Create a temporary file for the audio output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Generate audio with Dia
        logger.info(f"Generating TTS with Dia for text: '{text[:50]}...' with speed {speed}")
        start_time = time.time()
        
        # Handle SSML by stripping tags if present
        if "<speak" in text:
            logger.info("SSML detected in input text, stripping tags for Dia compatibility")
            # Simple regex to remove SSML tags (not perfect but works for basic cases)
            import re
            text = re.sub(r'<[^>]+>', '', text)
            text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            logger.info(f"Stripped text: '{text[:50]}...'")
        
        # Move model to the correct device if needed
        if dia_model.device != dia_device and torch.cuda.is_available():
            try:
                logger.info(f"Moving Dia model from {dia_model.device} to {dia_device}")
                dia_model.to(dia_device)
                logger.info(f"Dia model now on {dia_model.device}")
            except Exception as e:
                logger.error(f"Error moving model to {dia_device}: {e}")
        
        # Use mixed precision for faster inference on GPU
        with torch.inference_mode():
            # Use torch.amp.autocast instead of the deprecated torch.cuda.amp.autocast
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Generate audio with Dia using GPU acceleration if available
                logger.info(f"Generating audio with Dia on {dia_model.device}")
                start_gen_time = time.time()
                
                # Note: Dia doesn't directly use pitch, but we can adjust other parameters
                # Optimize generation parameters for speed
                audio_np = dia_model.generate(
                    text,
                    temperature=0.8,  # Lower temperature for faster, more deterministic generation
                    cfg_scale=2.0,    # Lower cfg_scale for faster generation
                    top_p=0.9,        # Slightly lower top_p for faster generation
                )
            
            gen_time = time.time() - start_gen_time
            logger.info(f"Audio generation completed in {gen_time:.2f} seconds")
        
        # Apply speed adjustment if needed
        if speed != 1.0 and audio_np is not None:
            try:
                # Load the audio with torchaudio
                sample_rate = 44100  # Dia's default sample rate
                
                # Convert to torch tensor
                audio_tensor = torch.tensor(audio_np).unsqueeze(0)  # Add channel dimension
                
                # Use torchaudio's speed function
                effects = [
                    ["speed", str(speed)],
                    ["rate", str(sample_rate)]
                ]
                audio_tensor, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                    audio_tensor, sample_rate, effects)
                
                # Convert back to numpy
                audio_np = audio_tensor.squeeze().numpy()
                logger.info(f"Successfully applied speed adjustment: {speed}")
            except Exception as speed_error:
                logger.warning(f"Failed to apply speed adjustment: {speed_error}")
                # Continue with original audio
        
        # Save the audio to the temporary file
        if audio_np is not None:
            sf.write(temp_path, audio_np, 44100)  # Dia uses 44.1kHz
            
            # Calculate duration in milliseconds
            duration_ms = int(len(audio_np) / 44100 * 1000)
            
            # Read the audio file
            with open(temp_path, "rb") as audio_file:
                audio_data = audio_file.read()
                
            logger.info(f"Dia TTS generation completed in {time.time() - start_time:.2f}s, duration: {duration_ms}ms")
            return audio_data, duration_ms
        else:
            logger.error("Dia TTS generation failed to produce audio output, falling back to Edge TTS")
            return await tts_edge(text, voice, speed)
            
    except Exception as e:
        logger.error(f"Error in Dia TTS generation: {e}, falling back to Edge TTS")
        return await tts_edge(text, voice, speed)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

async def tts_gtts(text: str, voice: str = "en", speed: float = 1.0) -> Tuple[bytes, int]:
    """Generate speech using Google Text-to-Speech"""
    try:
        from gtts import gTTS
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Run the TTS generation
        gtts_start_time = time.time()
        tts = gTTS(text=text, lang=voice[:2], slow=False)
        tts.save(temp_path)
        duration_ms = int((time.time() - gtts_start_time) * 1000)
        
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
async def tts_pyttsx3(text: str, voice: str = None, speed: float = 1.0, pitch: float = 1.0) -> Tuple[bytes, int]:
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
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Read the audio file
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        return audio_data, elapsed_ms
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
        if TTS_ENGINE == "edge" or TTS_ENGINE == "edge_tts" or TTS_ENGINE == "dia":
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

# Store background TTS tasks
background_tasks = {}

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest
) -> TTSResponse:
    """Synthesize speech from text using the specified TTS engine"""
    try:
        # Determine which TTS engine to use
        engine = request.engine.lower() if request.engine else TTS_ENGINE.lower()
        
        # Log the requested engine
        logger.info(f"Synthesizing speech with {engine} for text: '{request.text[:50]}...'")
        
        # If Dia was requested but we want to use Edge TTS by default
        if engine == "dia" and TTS_ENGINE.lower() == "edge":
            logger.info("Dia TTS requested but Edge TTS is configured as default. Using Edge TTS.")
            engine = "edge"
            
        logger.info(f"Using TTS engine: {engine}")
        
        # Get request parameters
        text = preprocess_text_for_tts(request.text)
        voice = request.voice or DEFAULT_VOICE
        speed = request.speed or 1.0
        pitch = request.pitch or 1.0
        
        # Track start time for performance measurement
        synthesis_start_time = time.time()
        
        # Select TTS engine
        audio_data = None
        duration_ms = 0
        audio_format = "wav"  # Default format
        
        # Process with appropriate TTS engine
        if engine == "edge" and EDGE_TTS_AVAILABLE:
            logger.info("Using Edge TTS engine")
            audio_data, duration_ms, audio_format = await tts_edge(text, voice, speed, pitch)
            
        elif engine == "dia" and DIA_AVAILABLE:
            logger.info("Using Dia TTS engine")
            # For longer text, process in background
            if len(text) > 100:
                logger.info(f"Processing long text with Dia TTS in background task: {len(text)} chars")
                task_id = str(uuid.uuid4())
                background_tasks[task_id] = {
                    "status": "processing",
                    "start_time": time.time(),
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch
                }
                
                # Start background task
                asyncio.create_task(process_dia_tts_background(task_id, text, voice, speed, pitch))
                
                # Return task ID for client to poll
                return TTSResponse(
                    audio_data="",
                    format="task",
                    duration_ms=0,
                    task_id=task_id,
                    status="processing"
                )
            else:
                # For shorter text, process synchronously
                audio_data, duration_ms = await tts_dia(text, voice, speed, pitch)
                audio_format = "wav"
                
        elif engine == "piper" and PIPER_AVAILABLE:
            logger.info("Using Piper TTS engine")
            audio_data, duration_ms = await tts_piper(text, voice, speed, pitch)
            audio_format = "wav"
            
        elif engine == "gtts" and GTTS_AVAILABLE:
            logger.info("Using gTTS engine")
            audio_data, duration_ms = await tts_gtts(text, voice, speed, pitch)
            audio_format = "mp3"
            
        elif engine == "pyttsx3" and PYTTSX3_AVAILABLE:
            logger.info("Using pyttsx3 engine")
            audio_data, duration_ms = await tts_pyttsx3(text, voice, speed, pitch)
            audio_format = "wav"
            
        else:
            # Fallback logic - try Edge TTS first, then Dia, then pyttsx3
            logger.warning(f"Unknown TTS engine: {engine}, trying fallback options")
            
            if EDGE_TTS_AVAILABLE:
                logger.info("Falling back to Edge TTS")
                audio_data, duration_ms = await tts_edge(text, voice, speed, pitch)
                audio_format = "wav"
            elif DIA_AVAILABLE:
                logger.info("Edge TTS not available, falling back to Dia TTS")
                audio_data, duration_ms = await tts_dia(text, voice, speed, pitch)
                audio_format = "wav"
            else:
                logger.warning("Edge TTS and Dia TTS not available, falling back to pyttsx3")
                audio_data, duration_ms = await tts_pyttsx3(text, voice, speed, pitch)
                audio_format = "wav"
        
        # Convert to base64
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Calculate total time
        total_time = time.time() - synthesis_start_time
        logger.info(f"TTS synthesis completed in {total_time:.2f}s for {len(text)} chars")
        
        return TTSResponse(
            audio_data=base64_audio,
            format=audio_format,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Error in TTS synthesis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    # Force Edge TTS as the default engine
    global TTS_ENGINE
    TTS_ENGINE = "edge"
    os.environ["TTS_ENGINE"] = "edge"
    logger.info(f"Starting TTS service with engine: {TTS_ENGINE} (forced to Edge TTS)")
    
    # Check if required packages are installed
    try:
        import edge_tts
        logger.info("edge-tts package found and will be used as primary TTS engine")
    except ImportError:
        logger.warning("edge-tts package not found. Install with: pip install edge-tts")
        logger.warning("Falling back to alternative TTS engines")
        TTS_ENGINE = "pyttsx3"
    
    # Check for other TTS engines as fallbacks
    try:
        from gtts import gTTS
        logger.info("gTTS package found (fallback available)")
    except ImportError:
        logger.warning("gTTS package not found. Install with: pip install gtts")
    
    # Always check for fallback TTS
    try:
        import pyttsx3
        logger.info("pyttsx3 package found (fallback TTS available)")
    except ImportError:
        logger.warning("pyttsx3 package not found. Install with: pip install pyttsx3")

    # Check if DEFAULT_VOICE is valid for Edge TTS
    if EDGE_TTS_AVAILABLE:
        try:
            available_voices = await edge_tts.list_voices()
            if not any(v['ShortName'] == DEFAULT_VOICE for v in available_voices):
                logger.warning(f"DEFAULT_VOICE '{DEFAULT_VOICE}' not found in available Edge TTS voices. "
                               f"TTS might fail or use a fallback voice. "
                               f"Consider using a voice from `edge-tts --list-voices` (e.g., en-US-JennyNeural).")
            else:
                logger.info(f"Default TTS voice '{DEFAULT_VOICE}' is valid for Edge TTS.")
        except Exception as e:
            logger.warning(f"Could not verify DEFAULT_VOICE for Edge TTS: {e}")
        
    logger.info("TTS service started successfully")

# Background TTS processing
async def background_tts_dia(task_id: str, text: str, voice: str, speed: float, pitch: float):
    """
    Process Dia TTS in background and store result for later retrieval
    """
    global background_tasks
    try:
        logger.info(f"Starting background TTS task {task_id}")
        background_tasks[task_id] = {
            "status": "processing",
            "text": text[:50] + "...",
            "start_time": time.time(),
            "progress": 0
        }
        
        # Generate audio with Dia
        audio_data, duration_ms = await tts_dia(text, voice, speed, pitch)
        
        # Store result
        if audio_data:
            # Convert to base64 for storage
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            background_tasks[task_id] = {
                "status": "completed",
                "audio_data": base64_audio,
                "format": "wav",
                "duration_ms": duration_ms,
                "completion_time": time.time()
            }
            logger.info(f"Background TTS task {task_id} completed successfully")
        else:
            background_tasks[task_id] = {
                "status": "failed",
                "error": "No audio data generated",
                "completion_time": time.time()
            }
            logger.error(f"Background TTS task {task_id} failed: No audio data generated")
            
    except Exception as e:
        logger.error(f"Background TTS task {task_id} failed: {str(e)}")
        background_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "completion_time": time.time()
        }

# Task status endpoint
@app.get("/tts_task/{task_id}", response_model=TTSResponse)
async def get_tts_task(task_id: str):
    """
    Get the status or result of a background TTS task
    """
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    task = background_tasks[task_id]
    status = task.get("status")
    
    if status == "processing":
        # Calculate progress based on time elapsed (rough estimate)
        start_time = task.get("start_time", time.time())
        elapsed = time.time() - start_time
        # Assume Dia takes about 10 minutes for a typical request
        progress = min(95, int(elapsed / 600 * 100))  # Cap at 95%
        
        return TTSResponse(
            audio_data="",
            format="task",
            duration_ms=0,
            task_id=task_id,
            status="processing",
            progress=progress
        )
    elif status == "completed":
        return TTSResponse(
            audio_data=task.get("audio_data", ""),
            format=task.get("format", "wav"),
            duration_ms=task.get("duration_ms", 0),
            task_id=task_id,
            status="completed"
        )
    else:  # failed
        raise HTTPException(
            status_code=500, 
            detail=f"Task {task_id} failed: {task.get('error', 'Unknown error')}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)
