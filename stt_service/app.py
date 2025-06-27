import os
import logging
import subprocess
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .transcribe import transcribe_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the model path being used by the transcribe module
# This is just to ensure it's visible when the app starts,
# transcribe.py already logs this when it's imported.
from .transcribe import MODEL_PATH as WHISPER_MODEL_USED
logger.info(f"STT Service will use Whisper model: {WHISPER_MODEL_USED}")

app = FastAPI(
    title="Speech-to-Text (STT) Service",
    description="A FastAPI service that uses whisper.cpp to transcribe audio.",
    version="0.1.0"
)

# Create a temporary directory for uploaded files if it doesn't exist
TEMP_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "stt_service_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/transcribe/")
async def create_transcription(file: UploadFile = File(...) ):
    """
    Receives an audio file, saves it temporarily, transcribes it using whisper.cpp,
    and returns the transcription.
    """
    if not file.content_type.startswith("audio/"):
        logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Save the uploaded file to a temporary location
        # Giving it a unique name to avoid collisions
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{file.filename}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Audio file '{file.filename}' saved temporarily to {temp_file_path}")

        # If the uploaded file is .webm, convert to .wav for whisper.cpp
        file_ext = os.path.splitext(temp_file_path)[1].lower()
        wav_file_path = None
        if file_ext == '.webm':
            wav_file_path = temp_file_path[:-5] + '.wav'
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', temp_file_path,
                '-ar', '16000', '-ac', '1',
                '-sample_fmt', 's16',
                '-map_metadata', '-1',
                wav_file_path
            ]
            try:
                logger.info(f"Converting .webm to .wav using ffmpeg: {' '.join(ffmpeg_cmd)}")
                ffmpeg_result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                logger.info(f"Conversion successful: {wav_file_path}")
                logger.info(f"ffmpeg stdout: {ffmpeg_result.stdout.decode(errors='ignore')}")
                logger.info(f"ffmpeg stderr: {ffmpeg_result.stderr.decode(errors='ignore')}")
            except Exception as e:
                logger.error(f"ffmpeg conversion failed: {e}")
                raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
        # Use the .wav file if converted, otherwise use the uploaded file
        audio_path_for_transcription = wav_file_path if wav_file_path else temp_file_path


        # Perform transcription
        transcription_text = transcribe_audio(audio_path_for_transcription)

        # Clean up the .wav file if it was created
        if wav_file_path and os.path.exists(wav_file_path):
            try:
                os.remove(wav_file_path)
                logger.info(f"Temporary wav file {wav_file_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temporary wav file {wav_file_path}: {e}")

        if transcription_text.startswith("Error:"):
            logger.error(f"Transcription failed for '{file.filename}': {transcription_text}")
            # Ensure the error message is JSON serializable
            return JSONResponse(
                status_code=500, 
                content={"error": "Transcription failed", "detail": transcription_text}
            )

        logger.info(f"Successfully transcribed '{file.filename}'")
        return {"filename": file.filename, "transcription": transcription_text}

    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing '{file.filename}': {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "An unexpected server error occurred.", "detail": str(e)}
        )
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} deleted.")
            except Exception as e_remove:
                logger.error(f"Error deleting temporary file {temp_file_path}: {e_remove}")
        # Ensure the file object from UploadFile is closed
        await file.close()

@app.get("/health")
async def health_check():
    """A simple health check endpoint."""
    return {"status": "STT Service is healthy"}

if __name__ == "__main__":
    import uvicorn
    # This is for local development/testing of this service independently
    # In a microservice setup, this would be run by a process manager or container orchestrator
    logger.info("Starting STT Service independently for development...")
    uvicorn.run(app, host="0.0.0.0", port=8001) # Using a different port, e.g., 8001 for STT service
