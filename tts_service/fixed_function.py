@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest
) -> TTSResponse:
    """Synthesize speech from text using the specified TTS engine"""
    try:
        # Determine which TTS engine to use
        engine = request.engine.lower() if request.engine else TTS_ENGINE.lower()
        
        # Log the requested engine
        logger.info(f"Synthesizing speech with {engine} for text: '{request.text[:50]}...'")
        
        # If Dia was requested but we want to use Edge TTS by default
        if engine == "dia" and TTS_ENGINE.lower() == "edge":
            logger.info("Dia TTS requested but Edge TTS is configured as default. Using Edge TTS.")
            engine = "edge"
            
        logger.info(f"Using TTS engine: {engine}")
        
        # Get request parameters
        text = request.text
        voice = request.voice or DEFAULT_VOICE
        speed = request.speed or 1.0
        pitch = request.pitch or 1.0
        
        # Track start time for performance measurement
        synthesis_start_time = time.time()
        
        # Select TTS engine
        audio_data = None
        duration_ms = 0
        audio_format = "wav"  # Default format
        
        # Process with appropriate TTS engine
        if engine == "edge" and EDGE_TTS_AVAILABLE:
            logger.info("Using Edge TTS engine")
            audio_data, duration_ms = await tts_edge(text, voice, speed, pitch)
            audio_format = "wav"
            
        elif engine == "dia" and DIA_AVAILABLE:
            logger.info("Using Dia TTS engine")
            # For longer text, process in background
            if len(text) > 100:
                logger.info(f"Processing long text with Dia TTS in background task: {len(text)} chars")
                task_id = str(uuid.uuid4())
                background_tasks[task_id] = {
                    "status": "processing",
                    "start_time": time.time(),
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch
                }
                
                # Start background task
                asyncio.create_task(process_dia_tts_background(task_id, text, voice, speed, pitch))
                
                # Return task ID for client to poll
                return TTSResponse(
                    audio_data="",
                    format="task",
                    duration_ms=0,
                    task_id=task_id,
                    status="processing"
                )
            else:
                # For shorter text, process synchronously
                audio_data, duration_ms = await tts_dia(text, voice, speed, pitch)
                audio_format = "wav"
                
        elif engine == "piper" and PIPER_AVAILABLE:
            logger.info("Using Piper TTS engine")
            audio_data, duration_ms = await tts_piper(text, voice, speed, pitch)
            audio_format = "wav"
            
        elif engine == "gtts" and GTTS_AVAILABLE:
            logger.info("Using gTTS engine")
            audio_data, duration_ms = await tts_gtts(text, voice, speed, pitch)
            audio_format = "mp3"
            
        elif engine == "pyttsx3" and PYTTSX3_AVAILABLE:
            logger.info("Using pyttsx3 engine")
            audio_data, duration_ms = await tts_pyttsx3(text, voice, speed, pitch)
            audio_format = "wav"
            
        else:
            # Fallback logic - try Edge TTS first, then Dia, then pyttsx3
            logger.warning(f"Unknown TTS engine: {engine}, trying fallback options")
            
            if EDGE_TTS_AVAILABLE:
                logger.info("Falling back to Edge TTS")
                audio_data, duration_ms = await tts_edge(text, voice, speed, pitch)
                audio_format = "wav"
            elif DIA_AVAILABLE:
                logger.info("Edge TTS not available, falling back to Dia TTS")
                audio_data, duration_ms = await tts_dia(text, voice, speed, pitch)
                audio_format = "wav"
            else:
                logger.warning("Edge TTS and Dia TTS not available, falling back to pyttsx3")
                audio_data, duration_ms = await tts_pyttsx3(text, voice, speed, pitch)
                audio_format = "wav"
        
        # Convert to base64
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Calculate total time
        total_time = time.time() - synthesis_start_time
        logger.info(f"TTS synthesis completed in {total_time:.2f}s for {len(text)} chars")
        
        return TTSResponse(
            audio_data=base64_audio,
            format=audio_format,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Error in TTS synthesis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")
