import os
import sys
import logging
import httpx
import time
import base64
import traceback
from fastapi import WebSocket
from typing import Dict, Any

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://localhost:8001/transcribe/")

# Import process_text function
from text_processor import process_text

async def _send_transcription_error(websocket: WebSocket, message: str):
    """Sends a standardized transcription error message."""
    try:
        await websocket.send_json({
            "type": "transcription_error",
            "text": message,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Failed to send transcription error to websocket: {e}")

async def process_audio(websocket: WebSocket, payload: Dict[str, Any]):
    """Process audio payload by decoding, sending to STT, and then to the text processor."""
    audio_b64 = payload.get("audio")
    history = payload.get("history", [])

    if not audio_b64:
        logger.error("Received audio message with no audio data.")
        await _send_transcription_error(websocket, "Empty audio payload received by server.")
        return

    try:
        # Decode the Base64 audio data
        audio_data = base64.b64decode(audio_b64)
        logger.info(f"Decoded audio data: {len(audio_data)} bytes")

        if len(audio_data) < 1000:  # 1KB threshold
            logger.warning("Audio data too small, likely not valid audio.")
            await _send_transcription_error(websocket, "Audio recording too short.")
            return
        
        # Send to STT service
        async with httpx.AsyncClient() as client:
            # The frontend sends audio/webm, which is what the STT service expects
            files = {'file': ('audio.webm', audio_data, 'audio/webm')}
            logger.info(f"Sending audio to STT service at {STT_SERVICE_URL}")
            
            response = await client.post(STT_SERVICE_URL, files=files, timeout=60.0)
            response.raise_for_status()
            
            stt_response = response.json()
            logger.info(f"STT response: {stt_response}")
            
            transcription = stt_response.get("transcription", "").strip()

            if transcription:
                logger.info(f"Transcription: '{transcription}'")
                
                # Send transcription result back to client for immediate UI feedback
                await websocket.send_json({
                    "type": "transcription_result",
                    "text": transcription,
                    "timestamp": time.time()
                })
                
                # Construct the payload for the text processor, including the original history
                text_payload = {
                    "text": transcription,
                    "history": history 
                }
                
                # Process the transcribed text with the LLM via the text_processor
                await process_text(websocket, text_payload)
            else:
                logger.warning("STT service returned an empty transcription.")
                await _send_transcription_error(websocket, "Could not understand audio.")

    except base64.binascii.Error as e:
        logger.error(f"Base64 decode error: {e}")
        await _send_transcription_error(websocket, "Invalid audio data format received.")
    except httpx.HTTPStatusError as e:
        logger.error(f"STT service HTTP error: {e.response.status_code} - {e.response.text}")
        await _send_transcription_error(websocket, f"Speech-to-text service error ({e.response.status_code}).")
    except httpx.RequestError as e:
        logger.error(f"STT service request error: {e}")
        await _send_transcription_error(websocket, "Could not connect to the speech-to-text service.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in process_audio: {e}", exc_info=True)
        traceback.print_exc()
        await _send_transcription_error(websocket, "An unexpected server error occurred while processing audio.")
