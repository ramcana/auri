import os
from dotenv import load_dotenv
import json
import logging
import httpx
import asyncio
from typing import Dict, List, Any, AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env before any os.getenv/os.environ calls
load_dotenv()

# Import the tool layer
from backend.tool_layer import get_real_world_context
from backend.chat import handle_message
# Updated memory_manager imports
from backend.memory_manager import summarize_facts, add_user_fact
# add_fact is deprecated, add_assistant_fact will be used in text_processor.py
from datetime import datetime # For timestamps

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables & Constants ---
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:11434')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '45'))

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Payloads ---
class GenerateRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = True

# --- Core LLM Interaction Logic ---

async def stream_llm_chat(payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    """
    Streams responses from the LLM's chat endpoint.
    """
    chat_url = f"{LLM_BASE_URL.rstrip('/')}/api/chat"
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            logger.info(f"Forwarding request to LLM at {chat_url}")
            async with client.stream("POST", chat_url, json=payload) as response:
                # Ensure we handle non-200 responses gracefully
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"LLM service returned error {response.status_code}: {error_body.decode()}")
                    error_payload = {
                        "error": f"LLM service failed with status {response.status_code}",
                        "details": error_body.decode()
                    }
                    yield json.dumps(error_payload).encode('utf-8') + b'\n'
                    return

                # Stream the successful response
                async for chunk in response.aiter_bytes():
                    yield chunk

    except httpx.RequestError as e:
        logger.error(f"Could not connect to LLM service at {chat_url}: {e}")
        error_payload = {
            "error": "Could not connect to the language model service."
        }
        yield json.dumps(error_payload).encode('utf-8') + b'\n'
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM stream: {e}")
        error_payload = {
            "error": "An unexpected error occurred while communicating with the LLM."
        }
        yield json.dumps(error_payload).encode('utf-8') + b'\n'


# --- API Endpoints ---

@app.post("/generate")
async def generate_endpoint(request: Request):
    """
    The main endpoint that the voice bot backend calls.
    It accepts a payload, checks for tool usage, and streams the LLM response.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON payload"}, 400

    # Basic validation
    if "messages" not in payload or "model" not in payload:
        return {"error": "Missing 'messages' or 'model' in request payload"}, 400

    # --- Tool Layer & User Profile Integration ---
    # Extract the last user message and user_id (if available)
    user_message_content = ""
    user_id = payload.get("user_id") or "default_user"
    if payload.get("messages") and payload["messages"][-1]["role"] == "user":
        user_message_content = payload["messages"][-1]["content"]

    # Log interaction and update user profile (tag/model can be extended)
    user_context = ""
    if user_message_content:
        # Optionally: tag detection could be improved with NLP or intent detection
        tag = None
        model = payload.get("model")
        user_context = handle_message(user_id, user_message_content, tag=tag, model=model)

        # --- Conversation Memory: Add structured user fact ---
        add_user_fact(user_id, user_message_content, timestamp=datetime.utcnow().isoformat())
        logger.info(f"Logged user fact for {user_id}: {user_message_content[:50]}...")

        # Inject summarized facts as a system message
        # summarize_facts will now return a string formatted for LLM consumption
        facts_summary = summarize_facts(user_id, max_turns=5) # Specify max_turns
        if facts_summary:
            logger.info(f"Injecting conversation facts: {facts_summary}")
            facts_message = {"role": "system", "content": facts_summary}
            payload["messages"].insert(-1, facts_message)

        # Call the tool layer to get any real-world context.
        context = get_real_world_context(user_message_content)
        
        # Inject user context (profile) as a system message before the user's message
        if user_context:
            logger.info(f"Injecting user profile context: {user_context}")
            profile_message = {"role": "system", "content": user_context}
            payload["messages"].insert(-1, profile_message)
        # Inject tool context (real-world context) as an additional system message
        if context:
            logger.info(f"Tool layer triggered. Injecting context: {context}")
            tool_message = {"role": "system", "content": context}
            payload["messages"].insert(-1, tool_message)

    logger.info(f"Received request for model '{payload.get('model')}' with {len(payload.get('messages', []))} messages.")

    return StreamingResponse(
        stream_llm_chat(payload),
        media_type="application/x-ndjson"
    )

@app.get("/")
def read_root():
    return {"status": "LLM service is running"}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # The port should be 8002 as expected by the other services
    uvicorn.run(app, host="0.0.0.0", port=8002)
