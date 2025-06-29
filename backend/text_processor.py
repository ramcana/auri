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
from datetime import datetime # For timestamps

# Import memory manager function
from backend.memory_manager import add_assistant_fact

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
    history = payload.get('history', []) # This history is from the client, might be different from server-side facts
    user_id = payload.get("user_id") or "default_user" # Assuming user_id might be in payload

    if not user_text.strip():
        logger.warning("Received empty text message, ignoring.")
        return

    # Note: context_history for the LLM prompt is built by llm_service.py using summarize_facts
    # The 'history' in this payload is the client-side view, which might not be used directly for LLM context here.
    # llm_service.py is responsible for constructing the full prompt including system messages, facts, and tool context.

    logger.info(f"Processing text from user {user_id}: '{user_text}'")
    
    stream_id = f"stream_{int(time.time() * 1000)}"

    # The llm_payload should ideally just be the user message and necessary parameters.
    # llm_service.py will enrich this with history and context.
    # However, the current structure has text_processor call llm_service with full message history.
    # For simplicity, we'll keep this structure for now.
    # The 'history' from payload is client-side, llm_service will use server-side summarized_facts.
    # The user_id needs to be passed to llm_service for it to manage facts correctly.

    # Construct messages for LLM service, ensuring user_id is passed for context management
    # The `llm_service` expects a "messages" list and will inject its own context (facts, tools)
    # The `history` in the payload here is the client's current view of the chat.
    # `llm_service.py` will use `summarize_facts(user_id)` for its historical context.

    # We will let llm_service.py handle the context building from summarize_facts(user_id).
    # We just need to send the current user message.
    # The payload to llm_service.py should include user_id.
    messages_for_llm = []
    if history: # Pass client history if available, llm_service might use it or parts of it
        messages_for_llm.extend(history) # This assumes history is already in {"role": ..., "content": ...} format
    messages_for_llm.append({"role": "user", "content": user_text})

    llm_payload = {
        "model": "llama3", # This could be made configurable
        "messages": messages_for_llm,
        "stream": True,
        "user_id": user_id # Pass user_id to llm_service
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

            # Log assistant's response to memory
            if full_response:
                # TODO: Determine if a tool was used by the LLM to generate this response.
                # This is tricky because tool_layer is called by llm_service *before* this point.
                # For now, we don't have direct info here if a tool was used by the LLM for this specific response.
                # llm_service.py would be a better place to log tool usage associated with a response,
                # but it only streams chunks.
                # A more robust solution would involve the LLM service sending structured events
                # about tool calls and final responses.
                # For now, just log the textual response.
                add_assistant_fact(
                    user_id=user_id,
                    content=full_response,
                    timestamp=datetime.utcnow().isoformat()
                )
                logger.info(f"Logged assistant fact for user {user_id}: {full_response[:50]}...")

    except httpx.RequestError as e:
        logger.error(f"Could not connect to LLM service: {e}")
        await send_error_message(websocket, "Could not connect to the language model service.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the LLM stream: {e}")
        traceback.print_exc()
        await send_error_message(websocket, "An unexpected error occurred while getting the response.")
