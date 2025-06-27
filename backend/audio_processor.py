import os
import sys
import logging
import httpx
import time
import base64
import traceback
import tempfile
import subprocess
import wave
import io # Added missing import
import webrtcvad # For Voice Activity Detection
from fastapi import WebSocket
from typing import Dict, Any, List, Tuple

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://localhost:8001/transcribe/")

# Import process_text function
from text_processor import process_text

# VAD Constants
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "3")) # 0 (least aggressive) to 3 (most aggressive)
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30")) # 10, 20, or 30
VAD_SAMPLE_RATE = 16000 # webrtcvad supports 8000, 16000, 32000, 48000
VAD_CHANNELS = 1
VAD_BYTES_PER_SAMPLE = 2 # 16-bit PCM

def _convert_webm_to_wav_p_cm(webm_data: bytes, target_sample_rate: int = VAD_SAMPLE_RATE, target_channels: int = VAD_CHANNELS) -> bytes:
    """Converts WEBM audio data to raw PCM WAV bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_webm_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
        tmp_webm_path = tmp_webm_file.name
        tmp_wav_path = tmp_wav_file.name

    try:
        with open(tmp_webm_path, 'wb') as f:
            f.write(webm_data)

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', tmp_webm_path,
            '-ar', str(target_sample_rate),
            '-ac', str(target_channels),
            '-sample_fmt', 's16le', # Signed 16-bit little-endian PCM
            '-map_metadata', '-1', # Remove metadata
            tmp_wav_path
        ]
        logger.info(f"Running ffmpeg for VAD pre-processing: {' '.join(ffmpeg_cmd)}")
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        if process.stderr:
             logger.debug(f"ffmpeg stderr (conversion to WAV for VAD): {process.stderr}")


        with open(tmp_wav_path, 'rb') as f:
            wav_data = f.read()
        return wav_data
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion to WAV for VAD failed: {e.stderr}")
        raise ValueError(f"ffmpeg conversion failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error converting WEBM to WAV for VAD: {e}")
        raise
    finally:
        if os.path.exists(tmp_webm_path):
            os.remove(tmp_webm_path)
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

def _apply_vad(wav_data: bytes) -> bytes:
    """Applies VAD to raw PCM WAV data and returns speech segments."""
    try:
        vad = webrtcvad.Vad()
        vad.set_mode(VAD_AGGRESSIVENESS)

        frame_duration_ms = VAD_FRAME_MS
        sample_rate = VAD_SAMPLE_RATE
        bytes_per_sample = VAD_BYTES_PER_SAMPLE
        channels = VAD_CHANNELS

        # Calculate frame size in bytes
        # For mono 16-bit PCM: (sample_rate * frame_duration_ms / 1000) * bytes_per_sample * channels
        frame_size_samples = int(sample_rate * frame_duration_ms / 1000)
        frame_size_bytes = frame_size_samples * bytes_per_sample * channels

        speech_frames = bytearray()

        # Open WAV data using wave module to read frames correctly
        with io.BytesIO(wav_data) as wav_io:
            with wave.open(wav_io, 'rb') as wf:
                if wf.getframerate() != sample_rate or wf.getnchannels() != channels or wf.getsampwidth() != bytes_per_sample:
                    logger.error(f"VAD input WAV format mismatch. Expected {sample_rate}Hz, {channels}ch, {bytes_per_sample}bps. Got {wf.getframerate()}Hz, {wf.getnchannels()}ch, {wf.getsampwidth()}bps.")
                    # Fallback to original data if format is unexpected, rather than failing hard
                    return wav_data

                num_frames_total = wf.getnframes()
                audio_content = wf.readframes(num_frames_total)

        voiced_segment = False
        for i in range(0, len(audio_content), frame_size_bytes):
            frame = audio_content[i:i+frame_size_bytes]
            if len(frame) < frame_size_bytes: # Skip partial frames at the end
                break
            try:
                is_speech = vad.is_speech(frame, sample_rate)
                if is_speech:
                    speech_frames.extend(frame)
                    if not voiced_segment: # Start of a voiced segment
                        voiced_segment = True
                        logger.debug("VAD: Speech segment started.")
                elif voiced_segment : # End of a voiced segment (if previously voiced)
                     voiced_segment = False
                     logger.debug("VAD: Speech segment ended.")

            except webrtcvad.Error as e:
                # This can happen if a frame is not of the correct length (e.g. 10, 20, 30ms)
                # or if sample rate is not supported. We already check sample rate.
                logger.warning(f"VAD error processing frame: {e}. Frame length: {len(frame)}")
                # As a fallback, append the frame to avoid losing audio
                speech_frames.extend(frame)


        if not speech_frames:
            logger.warning("VAD found no speech in audio, returning original audio.")
            return wav_data # Return original if no speech detected to avoid sending empty audio

        logger.info(f"VAD applied. Original size: {len(wav_data)} bytes, VAD output: {len(speech_frames)} bytes.")

        # Reconstruct a WAV file from the speech frames
        with io.BytesIO() as output_io:
            with wave.open(output_io, 'wb') as wf_out:
                wf_out.setnchannels(channels)
                wf_out.setsampwidth(bytes_per_sample)
                wf_out.setframerate(sample_rate)
                wf_out.writeframes(speech_frames)
            return output_io.getvalue()

    except ImportError:
        logger.warning("webrtcvad library not found. Skipping VAD.")
        return wav_data # Return original if VAD lib is missing
    except Exception as e:
        logger.error(f"Error during VAD processing: {e}", exc_info=True)
        return wav_data # Fallback to original data on any VAD error

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

        # 1. Convert original WebM audio to WAV for VAD processing
        logger.info("Converting WebM to WAV for VAD pre-processing...")
        try:
            wav_audio_data = _convert_webm_to_wav_p_cm(audio_data)
            logger.info(f"Converted to WAV for VAD: {len(wav_audio_data)} bytes")
        except ValueError as e: # Catch conversion errors
            logger.error(f"Audio conversion for VAD failed: {e}")
            await _send_transcription_error(websocket, "Audio conversion failed before VAD.")
            return
        except Exception as e: # Catch any other unexpected errors during conversion
            logger.error(f"Unexpected error during audio conversion for VAD: {e}", exc_info=True)
            await _send_transcription_error(websocket, "Unexpected server error during audio conversion.")
            return

        # 2. Apply VAD to the WAV data
        logger.info("Applying VAD...")
        vad_processed_audio_data = _apply_vad(wav_audio_data)
        
        if not vad_processed_audio_data or len(vad_processed_audio_data) < 1000: # Check if VAD output is too small
            logger.warning("VAD processing resulted in very small or empty audio. Sending original audio to STT.")
            # Fallback: use the original webm data if VAD fails or results in empty audio
            # The STT service can convert WebM to WAV itself.
            # This ensures we don't break transcription if VAD has issues.
            # However, our _apply_vad already returns original on error, so this is a double check.
            # For STT service, we will send the webm data if VAD output is not good.
            # If VAD was successful, vad_processed_audio_data is WAV.
            # The STT service needs to know the format.
            if len(vad_processed_audio_data) < 1000: # If VAD output is still bad
                 files = {'file': ('audio.webm', audio_data, 'audio/webm')}
                 logger.info("Using original WebM for STT due to small VAD output.")
            else: # VAD output is WAV and seems okay
                 files = {'file': ('audio.wav', vad_processed_audio_data, 'audio/wav')}
                 logger.info("Using VAD-processed WAV for STT.")

        else: # VAD output is WAV and seems okay
            files = {'file': ('audio.wav', vad_processed_audio_data, 'audio/wav')}
            logger.info("Using VAD-processed WAV for STT.")

        # 3. Send to STT service
        async with httpx.AsyncClient() as client:
            logger.info(f"Sending audio data to STT service at {STT_SERVICE_URL}")
            # `files` is now prepared with either original .webm or VAD-processed .wav
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
