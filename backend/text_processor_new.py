import os
import json
import logging
import httpx
import time
import asyncio
import re
from fastapi import WebSocket
from typing import Dict, Any, Union, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:11434/api/generate")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8003/synthesize")

# TTS settings
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en-US-AriaNeural")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_PITCH = float(os.getenv("TTS_PITCH", "1.0"))

async def send_to_tts(client: httpx.AsyncClient, websocket: WebSocket, text: str, part_id: int = 0) -> bool:
    """Send text to TTS service and forward audio to client"""
    try:
        # Clean text for TTS
        clean_text = text.strip()
        if not clean_text:
            return False
            
        logger.info(f"Sending text to TTS service (part {part_id}): {clean_text[:30]}...")
        
        # Send text to TTS service with enhanced parameters
        tts_response = await client.post(
            TTS_SERVICE_URL,
            json={
                "text": clean_text,
                "voice": DEFAULT_VOICE,
                "speed": TTS_SPEED,
                "pitch": TTS_PITCH
            },
            timeout=30.0
        )
        
        if tts_response.status_code == 200:
            try:
                # Parse the JSON response
                response_data = tts_response.json()
                
                # The TTS service returns base64-encoded audio data
                audio_data = response_data.get("audio_data", "")
                audio_format = response_data.get("format", "mp3")
                
                logger.info(f"Received TTS audio part {part_id} in {audio_format} format, length: {len(audio_data)} chars")
                
                # Send audio to client
                await websocket.send_json({
                    "type": "tts_audio",
                    "format": audio_format,
                    "data": audio_data,  # Already base64-encoded
                    "part": part_id,
                    "timestamp": time.time()
                })
                logger.info(f"TTS audio part {part_id} sent to client successfully")
                return True
            except Exception as e:
                logger.error(f"Error parsing TTS response: {str(e)}")
                return False
        else:
            error_detail = ""
            try:
                error_data = tts_response.json()
                error_detail = error_data.get("detail", tts_response.text)
            except Exception:
                error_detail = tts_response.text
                
            logger.error(f"TTS service error: {tts_response.status_code} - {error_detail}")
            return False
    except Exception as e:
        logger.error(f"Error with TTS service: {str(e)}")
        return False

# Helper function to check if text forms a complete sentence
def is_sentence_complete(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation and has minimum length"""
    return any(text.rstrip().endswith(end) for end in ['.', '!', '?', ':', ';']) and len(text) > 20

# Helper function to split text into sentences
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for better TTS chunking"""
    # Simple regex-based sentence splitter
    sentences = re.split(r'(?<=[.!?:;])\s+', text.strip())
    return [s for s in sentences if s.strip()]

async def process_text(websocket: WebSocket, text_message: str):
    """Process a text message from the client"""
    try:
        # If text_message is already a string, use it directly
        if isinstance(text_message, str):
            # Try to parse as JSON
            try:
                text_data = json.loads(text_message)
                text_content = text_data.get("text", "")
            except json.JSONDecodeError:
                # If not valid JSON, use the raw text
                text_content = text_message
        else:
            # If text_message is already parsed JSON (dict), use it directly
            text_content = text_message
            
        logger.info(f"Processing text message: {text_content}")
        
        # Send the text to the LLM service
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Sending text to LLM service: {text_content}")
                    
                # Extract text content if it's a dict
                if isinstance(text_content, dict) and 'text' in text_content:
                    text_content = text_content.get('text', '')
                
                # Prepare request to Ollama API
                # Get model name from environment variable or use a default that's available on the system
                model_name = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")  # Using mistral:7b-instruct which is available
                
                # First try the chat endpoint which is preferred for newer models
                try:
                    chat_request = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful voice assistant. Keep responses concise and conversational."
                            },
                            {
                                "role": "user",
                                "content": text_content if isinstance(text_content, str) else text_content.get('text', '')
                            }
                        ],
                        "stream": True  # Enable streaming responses
                    }
                    
                    logger.info(f"Using Ollama model: {model_name} with chat endpoint")
                    
                    # Try the chat endpoint first with streaming
                    chat_url = "http://localhost:11434/api/chat"
                    
                    # Create a unique message ID for this streaming session
                    stream_id = f"stream-{time.time()}-{id(asyncio.current_task())}" 
                    
                    # Send stream start message
                    await websocket.send_json({
                        "type": "stream_start",
                        "id": stream_id,
                        "timestamp": time.time()
                    })
                    
                    # Initialize response tracking variables
                    full_response = ""
                    sentence_buffer = ""
                    tts_sent_count = 0
                    
                    # Stream the response from the LLM
                    async with client.stream("POST", chat_url, json=chat_request, timeout=60.0) as response_stream:
                        async for chunk in response_stream.aiter_bytes():
                            try:
                                # Parse the JSON chunk
                                chunk_data = json.loads(chunk)
                                message = chunk_data.get("message", {})
                                chunk_text = message.get("content", "")
                                
                                if chunk_text:
                                    # Add to full response and sentence buffer
                                    full_response += chunk_text
                                    sentence_buffer += chunk_text
                                    
                                    # Send incremental update to client
                                    await websocket.send_json({
                                        "type": "stream_update",
                                        "id": stream_id,
                                        "text": chunk_text,
                                        "timestamp": time.time()
                                    })
                                    
                                    # Check if we have a complete sentence for TTS
                                    if is_sentence_complete(sentence_buffer):
                                        await send_to_tts(client, websocket, sentence_buffer, tts_sent_count)
                                        tts_sent_count += 1
                                        sentence_buffer = ""
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse streaming response chunk")
                            except Exception as e:
                                logger.error(f"Error processing streaming chunk: {str(e)}")
                    
                    # Process any remaining text in the sentence buffer
                    if sentence_buffer.strip():
                        await send_to_tts(client, websocket, sentence_buffer, tts_sent_count)
                        tts_sent_count += 1
                    
                    # Send completion message with full text
                    await websocket.send_json({
                        "type": "stream_end",
                        "id": stream_id,
                        "text": full_response,
                        "full_text": full_response,
                        "timestamp": time.time()
                    })
                    
                    logger.info(f"Completed streaming response with {tts_sent_count} TTS segments")
                
                except Exception as chat_error:
                    logger.error(f"Error with chat endpoint: {str(chat_error)}")
                    
                    # Fallback to generate endpoint
                    try:
                        generate_request = {
                            "model": model_name,
                            "prompt": text_content if isinstance(text_content, str) else text_content.get('text', ''),
                            "stream": True
                        }
                        
                        logger.info("Falling back to generate endpoint")
                        
                        # Create a unique message ID for this streaming session
                        stream_id = f"stream-fallback-{time.time()}-{hash(str(text_content))}"
                        
                        # Initialize response tracking variables
                        full_response = ""
                        sentence_buffer = ""
                        tts_sent_count = 0
                        
                        # Send initial streaming message to client
                        await websocket.send_json({
                            "type": "stream_start",
                            "id": stream_id,
                            "timestamp": time.time()
                        })
                        
                        # Stream the response chunks
                        async with client.stream("POST", LLM_SERVICE_URL, json=generate_request, timeout=60.0) as response:
                            async for chunk in response.aiter_bytes():
                                try:
                                    chunk_data = json.loads(chunk)
                                    chunk_text = chunk_data.get("response", "")
                                    
                                    if chunk_text:
                                        # Add to full response and sentence buffer
                                        full_response += chunk_text
                                        sentence_buffer += chunk_text
                                        
                                        # Send incremental update to client
                                        await websocket.send_json({
                                            "type": "stream_update",
                                            "id": stream_id,
                                            "text": chunk_text,
                                            "timestamp": time.time()
                                        })
                                        
                                        # Check if we have a complete sentence for TTS
                                        if is_sentence_complete(sentence_buffer):
                                            await send_to_tts(client, websocket, sentence_buffer, tts_sent_count)
                                            tts_sent_count += 1
                                            sentence_buffer = ""
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse streaming response chunk")
                                except Exception as e:
                                    logger.error(f"Error processing streaming chunk: {str(e)}")
                        
                        # Process any remaining text in the sentence buffer
                        if sentence_buffer.strip():
                            await send_to_tts(client, websocket, sentence_buffer, tts_sent_count)
                        
                        # Send completion message
                        await websocket.send_json({
                            "type": "stream_end",
                            "id": stream_id,
                            "text": full_response,
                            "full_text": full_response,
                            "timestamp": time.time()
                        })
                        
                        logger.info(f"Completed fallback streaming response with {tts_sent_count} TTS segments")
                        
                    except Exception as generate_error:
                        logger.error(f"Error with generate endpoint: {str(generate_error)}")
                        raise generate_error
            
            except Exception as e:
                logger.error(f"Error processing text with LLM: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "text": "Error processing your message with the language model",
                    "timestamp": time.time()
                })
    
    except Exception as e:
        logger.error(f"Error in process_text: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "text": "Error processing your message",
                "timestamp": time.time()
            })
        except Exception as send_error:
            logger.error(f"Error sending error response: {str(send_error)}")
