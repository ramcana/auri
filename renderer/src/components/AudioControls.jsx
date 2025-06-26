import React, { useState, useEffect } from 'react';
import { stopAudioPlayback, clearAudioQueue, getAudioPlaybackState } from '../utils/audio';
import './AudioControls.css';

/**
 * Global audio controls component for the voice bot
 * Provides controls to stop audio playback and displays playback status
 */
const AudioControls = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentAudioInfo, setCurrentAudioInfo] = useState(null);
  
  // Check audio playback state periodically
  useEffect(() => {
    const checkInterval = setInterval(() => {
      const audioState = getAudioPlaybackState();
      setIsPlaying(audioState.isPlaying);
      
      if (audioState.isPlaying) {
        setCurrentAudioInfo({
          id: audioState.currentAudioId,
          part: audioState.currentPartId,
          format: audioState.format
        });
      } else {
        setCurrentAudioInfo(null);
      }
    }, 200);
    
    return () => clearInterval(checkInterval);
  }, []);
  
  // Handle stopping all audio playback
  const handleStopAudio = () => {
    stopAudioPlayback();
    clearAudioQueue();
  };
  
  // Only show controls when audio is playing
  if (!isPlaying) return null;
  
  return (
    <div className="global-audio-controls">
      <div className="audio-status">
        <span className="audio-indicator">🔊</span>
        <span>Playing audio</span>
      </div>
      <button 
        className="stop-audio-button" 
        onClick={handleStopAudio}
        title="Stop all audio playback"
      >
        <span role="img" aria-label="Stop">⏹️</span> Stop
      </button>
    </div>
  );
};

export default AudioControls;
