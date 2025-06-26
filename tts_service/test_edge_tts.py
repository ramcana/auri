import asyncio
import edge_tts
import sys

async def main():
    text = "Hello, this is a test of Microsoft Edge TTS."
    voice = "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    print(f"Generated {len(audio_data)} bytes of audio.")
    with open("test_edge_tts_output.wav", "wb") as f:
        f.write(audio_data)
    print("Audio written to test_edge_tts_output.wav")

if __name__ == "__main__":
    asyncio.run(main())
