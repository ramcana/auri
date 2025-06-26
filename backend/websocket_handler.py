import asyncio
import base64
import json
import time
import logging
import os
import sys
from typing import List, Dict, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import processing functions from the new modules
from text_processor import process_text
from audio_processor import process_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_handler")

# Store active connections
active_connections: List[WebSocket] = []

async def websocket_endpoint(websocket: WebSocket):
    """
    Handle WebSocket connections and message processing.
    """
    client_id = id(websocket)
    logger.info(f"⚪ New WebSocket connection attempt from client {client_id}")
    
    try:
        # Accept the WebSocket connection
        await websocket.accept()
        logger.info(f"🟢 WebSocket connection accepted for client {client_id}")
        
        # Send immediate connection acknowledgment
        await websocket.send_json({
            "type": "connection_ack",
            "status": "connected",
            "timestamp": time.time(),
            "message": "Successfully connected to Voice Bot Backend"
        })
        
        # Add to active connections
        active_connections.append(websocket)
        logger.info(f"🟢 Client {client_id} added to active connections. Total: {len(active_connections)}")
        
        # Begin receive loop
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                # Wait for any incoming message (text or bytes)
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    logger.info(f"Client {client_id} disconnected gracefully.")
                    break
                
                # All business logic messages are now text-based JSON
                if "text" in message and message["text"] is not None:
                    text_data = message["text"]
                    
                    try:
                        payload = json.loads(text_data)
                        msg_type = payload.get("type")

                        if msg_type == 'ping':
                            logger.debug(f"Received ping from {client_id}, sending pong.")
                            await websocket.send_json({"type": "pong", "timestamp": time.time()})
                        
                        elif msg_type == 'text':
                            logger.info(f"Received text message from {client_id}")
                            # Pass the entire payload to the processor
                            await process_text(websocket, payload)
                            
                        elif msg_type in ('audio', 'user_audio', 'stt_audio'):
                            logger.info(f"Received {msg_type} message from {client_id}")
                            # Pass the entire payload to the processor
                            await process_audio(websocket, payload)

                        else:
                            logger.warning(f"Received unknown message type: '{msg_type}'")

                    except json.JSONDecodeError:
                        logger.error(f"Could not decode JSON from message: {text_data[:200]}...")
                
                # The 'bytes' message type is no longer used for audio transmission
                elif "bytes" in message and message["bytes"] is not None:
                    logger.warning(f"Received unexpected binary message from {client_id} of size {len(message['bytes'])}. This is deprecated.")

            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected gracefully.")
                break # Exit loop
            
            except Exception as e:
                # Log the full traceback for unexpected errors
                logger.error(f"An unexpected error occurred in the receive loop for client {client_id}: {e}", exc_info=True)
                break # Exit loop on other errors
    
    except WebSocketDisconnect:
        logger.info(f"🔴 Client {client_id} disconnected during handshake")
    
    except Exception as e:
        logger.error(f"❌ Error in WebSocket connection: {str(e)}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except Exception:
            pass
    
    finally:
        # Clean up connection
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info(f"🔴 Removed client {client_id} from active connections. Total: {len(active_connections)}")
