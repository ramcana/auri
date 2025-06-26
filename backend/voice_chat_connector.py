import os
import json
import logging
import asyncio
import websockets
import httpx
from backend.conversation_manager import session_manager, ConversationManager
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
LLM_SERVICE_PORT = os.getenv("LLM_SERVICE_PORT", "8000")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", f"http://localhost:{LLM_SERVICE_PORT}")
LLM_WS_URL = os.getenv("LLM_WS_URL", f"ws://localhost:{LLM_SERVICE_PORT}/ws/chat")
TTS_PORT = os.getenv("TTS_PORT", "8003")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", f"http://localhost:{TTS_PORT}/synthesize")

# FastAPI app
app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store session IDs for clients
client_sessions: Dict[str, str] = {}

async def forward_to_tts(text: str, websocket: WebSocket):
    """Forward text to TTS service and send audio URL back to client"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TTS_SERVICE_URL,
                json={"text": text}
            )
            
            if response.status_code == 200:
                data = response.json()
                await websocket.send_json({
                    "type": "tts_audio",
                    "url": data.get("audio_url"),
                    "text": text
                })
            else:
                logger.error(f"TTS service error: {response.text}")
                await websocket.send_json({
                    "type": "tts_error",
                    "message": f"TTS service error: {response.status_code}"
                })
    except Exception as e:
        logger.error(f"Error forwarding to TTS: {e}")
        await websocket.send_json({
            "type": "tts_error",
            "message": f"Error forwarding to TTS: {str(e)}"
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint that handles voice bot communication"""
    await websocket.accept()
    client_id = str(id(websocket))
    active_connections[client_id] = websocket
    
    try:
        # Connect to LLM WebSocket service
        async with websockets.connect(LLM_WS_URL) as llm_ws:
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type", "")
                
                if message_type == "stt_result":
                    # Speech-to-text result from client
                    transcript = data.get("text", "")
                    if not transcript.strip():
                        continue
                    
                    # Get or create session ID
                    session_id = client_sessions.get(client_id)
                    if not session_id:
                        # First message, create a new session
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"{LLM_SERVICE_URL}/chat",
                                json={
                                    "message": "Hello",  # Initial message to create session
                                    "mode": "default"
                                }
                            )
                            if response.status_code == 200:
                                session_data = response.json()
                                session_id = session_data.get("session_id")
                                client_sessions[client_id] = session_id
                    
                    # Send message to LLM WebSocket
                    await llm_ws.send(json.dumps({
                        "session_id": session_id,
                        "message": transcript,
                        "mode": "default"
                    }))
                    
                    # Notify client that processing has started
                    await websocket.send_json({
                        "type": "processing_started",
                        "text": transcript
                    })
                    
                    # Process LLM responses
                    full_response = ""
                    while True:
                        llm_response = await llm_ws.recv()
                        llm_data = json.loads(llm_response)
                        
                        if "error" in llm_data:
                            await websocket.send_json({
                                "type": "error",
                                "message": llm_data["error"]
                            })
                            break
                            
                        if llm_data.get("type") == "chunk":
                            # Stream chunk to TTS if it ends with sentence-ending punctuation
                            chunk = llm_data.get("content", "")
                            full_response += chunk
                            
                            # Check if chunk ends with sentence-ending punctuation
                            if chunk.strip().endswith((".", "!", "?", ":", ";")) and len(chunk.strip()) > 5:
                                await forward_to_tts(chunk, websocket)
                                
                        elif llm_data.get("type") == "complete":
                            # Process any remaining text
                            if full_response:
                                await forward_to_tts(full_response, websocket)
                                
                            # Send completion message
                            await websocket.send_json({
                                "type": "response_complete",
                                "full_text": full_response
                            })
                            break
                
                elif message_type == "clear_memory":
                    # Clear conversation memory
                    session_id = client_sessions.get(client_id)
                    if session_id:
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                f"{LLM_SERVICE_URL}/chat/clear",
                                json={"session_id": session_id}
                            )
                        await websocket.send_json({
                            "type": "memory_cleared"
                        })
                
                elif message_type == "ping":
                    # Ping to keep connection alive
                    await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Clean up
        active_connections.pop(client_id, None)
        client_sessions.pop(client_id, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
