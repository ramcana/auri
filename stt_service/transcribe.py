import subprocess
import os
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine the base directory of this script to locate whisper.cpp and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_CPP_EXE = os.path.join(BASE_DIR, 'whisper-cli.exe')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ggml-base.en.bin')

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes the given audio file using whisper.cpp.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcribed text, or an error message if transcription fails.
    """
    if not os.path.exists(WHISPER_CPP_EXE):
        logger.error(f"Whisper.cpp executable not found at: {WHISPER_CPP_EXE}")
        return "Error: Whisper.cpp executable not found."
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Whisper.cpp model not found at: {MODEL_PATH}")
        return "Error: Whisper.cpp model not found."
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return "Error: Audio file not found."

    command = [
        WHISPER_CPP_EXE,
        "-m", MODEL_PATH,
        "-f", audio_file_path,
        "-nt",             # No timestamps
        "-otxt",           # Output as plain text
        # "-osrt", # Uncomment for SRT output
        # "-ovtt", # Uncomment for VTT output
    ]

    try:
        logger.info(f"Running whisper.cpp command: {' '.join(command)}")
        # whisper.cpp outputs text to stdout and .txt file in the same dir as input audio
        process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=os.path.dirname(audio_file_path))
        
        # The output text file will have the same name as the input audio file, but with .txt extension
        output_txt_file = os.path.splitext(audio_file_path)[0] + '.txt'
        
        if os.path.exists(output_txt_file):
            with open(output_txt_file, 'r', encoding='utf-8') as f:
                transcribed_text = f.read().strip()
            os.remove(output_txt_file) # Clean up the text file
            logger.info(f"Transcription successful: {transcribed_text}")
            return transcribed_text
        else:
            # Fallback to stdout if .txt file is not found (though whisper.cpp usually creates it)
            if process.stdout:
                logger.info(f"Transcription successful (from stdout): {process.stdout.strip()}")
                return process.stdout.strip()
            logger.error("Transcription failed: No output text file found and stdout is empty.")
            return "Error: Transcription failed to produce output."

    except subprocess.CalledProcessError as e:
        logger.error(f"whisper.cpp execution failed: {e}")
        # Diagnostic: dump first 200 bytes of the audio file as hex
        hex_dump = ''
        try:
            with open(audio_file_path, 'rb') as f:
                wav_bytes = f.read(200)
                hex_dump = wav_bytes.hex()
                logger.error(f"First 200 bytes of {audio_file_path}: {hex_dump}")
        except Exception as dump_err:
            logger.error(f"Failed to hex-dump {audio_file_path}: {dump_err}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Stderr: {e.stderr}")
            return f"Error: whisper.cpp execution failed. {e.stderr} HEXDUMP={hex_dump}"
        else:
            return f"Error: whisper.cpp execution failed. No stderr output. {str(e)} HEXDUMP={hex_dump}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during transcription: {e}")
        return f"Error: An unexpected error occurred. {e}"

if __name__ == '__main__':
    # Example usage: Create a dummy wav file and transcribe it
    # In a real scenario, you'd pass a path to an actual audio file.
    logger.info("Running example transcription...")
    # Create a temporary dummy audio file for testing
    # Note: whisper.cpp expects a valid WAV file. This dummy will cause an error.
    # Replace with a path to a real .wav file for actual testing.
    temp_dir = tempfile.mkdtemp()
    dummy_audio_file = os.path.join(temp_dir, "test_audio.wav")
    with open(dummy_audio_file, 'w') as f:
        f.write("dummy audio data") # This is not a valid WAV file
    
    logger.warning(f"Using a dummy audio file for testing: {dummy_audio_file}. This will likely fail unless whisper.cpp handles it gracefully.")
    logger.warning("Replace with a real .wav file to test transcription properly.")
    
    transcription = transcribe_audio(dummy_audio_file)
    print(f"Transcription Result:\n{transcription}")
    
    # Clean up dummy file and directory
    os.remove(dummy_audio_file)
    os.rmdir(temp_dir)
