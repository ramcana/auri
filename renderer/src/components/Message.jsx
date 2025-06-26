import React, { useState, useEffect, useRef } from 'react';
import './Message.css';
import { playAudioFromBase64, stopAudioPlayback, getAudioPlaybackState } from '../utils/audio';

// Placeholder for avatar images or icons
const Avatar = ({ avatar }) => <div className="avatar">{avatar}</div>;

// Audio control component with play/stop buttons
const AudioControls = ({ isPlaying, onPlay, onStop }) => (
  <div className="audio-controls">
    {isPlaying ? (
      <button className="audio-button stop" onClick={onStop} title="Stop playback">
        <span role="img" aria-label="Stop">⏹️</span>
      </button>
    ) : (
      <button className="audio-button play" onClick={onPlay} title="Play message">
        <span role="img" aria-label="Play">▶️</span>
      </button>
    )}
  </div>
);

const Message = ({ message }) => {
  // Extract message content from either text or data property
  // Backend sends 'data' for assistant messages, but might use 'text' for other messages
  const { id, sender, text, data, role, avatar, hasAudio, isError, audio, audio_format } = message;
  const messageContent = data || text || '';
  const isUser = sender === 'user' || role === 'user';
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAudioAvailable, setIsAudioAvailable] = useState(Boolean(audio));
  const audioCheckInterval = useRef(null);

  // Check if this message is currently playing audio
  useEffect(() => {
    // Start interval to check audio playback state
    audioCheckInterval.current = setInterval(() => {
      const audioState = getAudioPlaybackState();
      
      // If this message's ID matches the currently playing audio
      if (audioState.currentAudioId === id && audioState.isPlaying) {
        setIsPlaying(true);
      } else if (isPlaying && (!audioState.isPlaying || audioState.currentAudioId !== id)) {
        // If we think we're playing but we're not anymore
        setIsPlaying(false);
      }
    }, 200); // Check every 200ms
    
    return () => {
      // Clean up interval and audio playback when unmounting
      clearInterval(audioCheckInterval.current);
      if (isPlaying) {
        stopAudioPlayback();
      }
    };
  }, [id, isPlaying]);
  
  // Update audio availability when audio prop changes
  useEffect(() => {
    setIsAudioAvailable(Boolean(audio));
  }, [audio]);

  // Handle audio playback
  const handlePlayAudio = () => {
    if (audio) {
      setIsPlaying(true);
      playAudioFromBase64(audio, audio_format || 'mp3', 0, id)
        .catch(error => {
          console.error('Error playing audio:', error);
          setIsPlaying(false);
        });
    }
  };
  
  // Handle stopping audio playback
  const handleStopAudio = () => {
    if (stopAudioPlayback()) {
      setIsPlaying(false);
    }
  };

  // Add CSS classes based on message properties
  const bubbleClasses = [
    'message-bubble',
    isUser ? 'user-bubble' : 'bot-bubble',
    isError ? 'error-bubble' : '',
    isAudioAvailable ? 'has-audio' : '',
    isPlaying ? 'playing-audio' : ''
  ].filter(Boolean).join(' ');

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      {!isUser && <Avatar avatar={avatar || '🤖'} />}
      <div className={bubbleClasses}>
        {isAudioAvailable && 
          <AudioControls 
            isPlaying={isPlaying} 
            onPlay={handlePlayAudio}
            onStop={handleStopAudio}
          />}
        <p className="message-text">{messageContent}</p>
        {isPlaying && <div className="audio-indicator">🔊 Playing audio...</div>}
      </div>
      {isUser && <Avatar avatar={avatar || '👤'} />}
    </div>
  );
};

export default Message;
