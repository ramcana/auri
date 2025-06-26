import os
import json
import logging
import httpx
import time
import asyncio
import re
import traceback
from fastapi import WebSocket
from typing import Dict, Any, List, Optional

# --- Helper Functions ---

async def send_error_message(websocket: WebSocket, message: str):
    """Sends a standardized error message to the client."""
    try:
        await websocket.send_json({
            "type": "error",
            "message": message,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Failed to send error message to websocket: {e}")

# --- Logging Configuration ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables & Constants ---

LLM_SERVICE_URL = os.getenv('LLM_SERVICE_URL', 'http://localhost:8002/generate')
TTS_SERVICE_URL = os.getenv('TTS_SERVICE_URL', 'http://localhost:8003/synthesize')
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en-US-JennyNeural")
os.environ['TTS_ENGINE'] = 'edge'  # Force Edge TTS for performance
TTS_ENGINE = 'edge'

def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))

TTS_SPEED = clamp(float(os.getenv("TTS_SPEED", "1.0")), 0.5, 2.0)
TTS_PITCH = clamp(float(os.getenv("TTS_PITCH", "1.0")), 0.5, 2.0)
logger.info(f"Using TTS engine: {TTS_ENGINE} (forced)")

# --- Text Cleaning for TTS ---

def clean_for_tts(text: str) -> str:
    """Cleans text to be more suitable for Text-to-Speech synthesis."""
    if not text:
        return ""
    
    text = re.sub(r'[\*`#_]', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = text.replace('\n', ' ').replace('  ', ' ').strip()
    
    return text

# --- TTS Interaction ---

async def send_to_tts(client: httpx.AsyncClient, text: str, part_id: str, message_id: str) -> Optional[Dict[str, Any]]:
    """Sends text to the TTS service and returns the audio data payload."""
    cleaned_text = clean_for_tts(text)
    if not cleaned_text:
        logger.warning("Skipping TTS for empty or cleaned-out text.")
        return None

    logger.info(f"Requesting TTS for part {part_id}: '{cleaned_text[:60]}...'")
    
    try:
        response = await client.post(
            TTS_SERVICE_URL,
            json={
                "text": cleaned_text,
                "voice": DEFAULT_VOICE,
                "speed": TTS_SPEED,
                "pitch": TTS_PITCH,
                "engine": TTS_ENGINE
            },
            timeout=20.0
        )
        response.raise_for_status()
        response_data = response.json()
        audio_data_b64 = response_data.get("audio_data")

        if audio_data_b64:
            logger.info(f"TTS audio part {part_id} received for message {message_id}")
            return {
                "type": "tts_audio",
                "audio": audio_data_b64,
                "format": response_data.get("format", "wav"),
                "part_id": part_id,
                "message_id": message_id,
                "timestamp": time.time()
            }
        else:
            logger.warning("TTS service returned no audio data.")
            return None

    except httpx.RequestError as e:
        logger.error(f"TTS request failed for part {part_id}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"TTS service returned an error for part {part_id}: {e.response.status_code} - {e.response.text}")
        return None


# --- Sentence Processing ---

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for better TTS chunking."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.?!])\s|(?<=[.?!])$', text.strip())
    return [s.strip() for s in sentences if s and s.strip()]

# --- Main Text Processing Logic ---

async def process_text(websocket: WebSocket, payload: Dict[str, Any]):
    """Process a text message from the client, using conversation history for context."""
    user_text = payload.get('text', '')
    history = payload.get('history', [])

    if not user_text.strip():
        logger.warning("Received empty text message, ignoring.")
        return

    context_history = history[:-1] if history else []

    logger.info(f"Processing text: '{user_text}' with {len(context_history)} context messages.")
    
    stream_id = f"stream_{int(time.time() * 1000)}"
    llm_payload = {
        "model": "llama3",
        "messages": [
            *context_history,
            {"role": "user", "content": user_text}
        ],
        "stream": True
    }

    full_response = ""
    sentence_buffer = ""
    tts_tasks = []
    tts_sent_count = 0
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Connecting to LLM service at {LLM_SERVICE_URL}")
            async with client.stream("POST", LLM_SERVICE_URL, json=llm_payload, timeout=45.0) as response:
                if response.status_code != 200:
                    response_bytes = await response.aread()
                    response_text = response_bytes.decode('utf-8')
                    logger.error(f"LLM service returned an error: {response.status_code} - {response_text}")
                    await send_error_message(websocket, f"The language model service failed with status {response.status_code}.")
                    return

                await websocket.send_json({
                    "type": "stream_start",
                    "message_id": stream_id,
                    "timestamp": time.time()
                })
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        
                        if chunk:
                            full_response += chunk
                            sentence_buffer += chunk
                            
                            await websocket.send_json({
                                "type": "text_chunk",
                                "text": chunk,
                                "message_id": stream_id,
                                "timestamp": time.time()
                            })
                            
                            sentences = split_into_sentences(sentence_buffer)
                            
                            if len(sentences) > 1:
                                sentences_to_process = sentences[:-1]
                                for sentence in sentences_to_process:
                                    if sentence:
                                        tts_part_id = f"{stream_id}_{tts_sent_count}"
                                        task = asyncio.create_task(send_to_tts(client, sentence, tts_part_id, message_id=stream_id))
                                        tts_tasks.append(task)
                                        tts_sent_count += 1
                                
                                sentence_buffer = sentences[-1]

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response chunk: {line}")
                    except Exception as e:
                        logger.error(f"Error processing streaming chunk: {str(e)}")
            
            if sentence_buffer.strip():
                logger.info(f"Processing final buffer part for TTS: '{sentence_buffer.strip()[:50]}...'")
                tts_part_id = f"{stream_id}_{tts_sent_count}"
                task = asyncio.create_task(send_to_tts(client, sentence_buffer.strip(), tts_part_id, message_id=stream_id))
                tts_tasks.append(task)
                tts_sent_count += 1

            if tts_tasks:
                logger.info(f"Waiting for {len(tts_tasks)} TTS tasks to complete...")
                audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
                
                tts_error_occurred = False
                for result in audio_results:
                    if isinstance(result, Exception):
                        logger.error(f"A TTS task failed: {result}")
                        tts_error_occurred = True
                    elif result:
                        try:
                            await websocket.send_json(result)
                            logger.info(f"Sent audio part {result.get('part_id')} to client.")
                        except Exception as e:
                            logger.error(f"Failed to send audio data to client: {e}")
                            break
                    else:
                        logger.warning("A TTS task returned no audio data.")
                        tts_error_occurred = True
                
                if tts_error_occurred:
                    await send_error_message(websocket, "An error occurred during speech synthesis. Some audio may be missing.")
        
            await websocket.send_json({
                "type": "stream_end",
                "message_id": stream_id,
                "full_response": full_response,
                "timestamp": time.time()
            })
            logger.info(f"Stream {stream_id} ended. Completed with {tts_sent_count} TTS segments")

    except httpx.RequestError as e:
        logger.error(f"Could not connect to LLM service: {e}")
        await send_error_message(websocket, "Could not connect to the language model service.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the LLM stream: {e}")
        traceback.print_exc()
        await send_error_message(websocket, "An unexpected error occurred while getting the response.")
