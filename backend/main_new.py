import os
import logging
import json
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# STT service URL (local whisper.cpp microservice)
STT_SERVICE_URL = "http://localhost:8001/transcribe/"

# Create FastAPI app
app = FastAPI(
    title="Voice Bot Backend",
    description="Handles WebSocket connections for voice and text input processing",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}


async def process_audio(websocket: WebSocket, audio_data: bytes):
    """Process audio data by sending it to the STT service"""
    try:
        # Send to STT service
        async with httpx.AsyncClient() as client:
            files = {'file': ('audio.wav', audio_data, 'audio/wav')}
            logger.info(f"Sending audio to STT service at {STT_SERVICE_URL}")
            
            response = await client.post(STT_SERVICE_URL, files=files, timeout=60.0)
            response.raise_for_status()
            
            stt_response = response.json()
            logger.info(f"STT response: {stt_response}")
            
            if "transcription" in stt_response:
                await websocket.send_json({
                    "type": "transcription_result",
                    "text": stt_response["transcription"]
                })
            else:
                await websocket.send_json({
                    "type": "transcription_error",
                    "text": "Failed to transcribe audio"
                })
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        await websocket.send_json({
            "type": "transcription_error",
            "text": f"STT service error: {e.response.status_code}"
        })
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        await websocket.send_json({
            "type": "transcription_error",
            "text": "Could not connect to STT service"
        })
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        await websocket.send_json({
            "type": "transcription_error",
            "text": "Error processing audio"
        })


async def process_text(websocket: WebSocket, text_message: str):
    """Process text messages from the client"""
    try:
        # Parse the JSON message
        text_data = json.loads(text_message)
        
        if text_data.get("type") == "text":
            text_content = text_data.get("content", "")
            logger.info(f"Processing text: {text_content}")
            
            # Echo back the text (placeholder for NLP/chatbot integration)
            await websocket.send_json({
                "type": "transcription_result", 
                "text": text_content
            })
        else:
            logger.warning(f"Unknown message type: {text_data.get('type')}")
            await websocket.send_json({
                "type": "transcription_error", 
                "text": "Unknown message type"
            })
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON: {text_message}")
        await websocket.send_json({
            "type": "transcription_error", 
            "text": "Invalid message format"
        })
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        await websocket.send_json({
            "type": "transcription_error",
            "text": "Error processing text message"
        })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint that handles both audio and text messages"""
    logger.info("WebSocket connection established")
    await websocket.accept()
    
    try:
        while True:
            try:
                # Receive message from client
                data = await websocket.receive()
                
                # Handle binary audio data
                if "bytes" in data:
                    audio_data = data["bytes"]
                    logger.info(f"Received {len(audio_data)} bytes of audio data")
                    await process_audio(websocket, audio_data)
                
                # Handle text messages
                elif "text" in data:
                    await process_text(websocket, data["text"])
                
                # Unknown message format
                else:
                    logger.warning("Received unknown data format")
                    await websocket.send_json({
                        "type": "error",
                        "text": "Unknown message format"
                    })
            
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            
            except Exception as e:
                logger.error(f"Error handling message: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "text": "Server error occurred"
                    })
                except Exception:
                    logger.error("Failed to send error response")
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"Fatal WebSocket error: {str(e)}")


# Run the app when executed directly
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Voice Bot Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
