import React, { useState, useEffect } from 'react';
import './VoiceButton.css';

/**
 * A central voice control button similar to Siri's interface
 * 
 * @param {Object} props Component props
 * @param {boolean} props.isRecording Whether recording is active
 * @param {boolean} props.isPushToTalk Whether in push-to-talk mode
 * @param {boolean} props.isConnected Whether connected to server
 * @param {boolean} props.isMicrophoneAvailable Whether microphone is available
 * @param {Function} props.onPushToTalkStart Function to call when push-to-talk starts
 * @param {Function} props.onPushToTalkEnd Function to call when push-to-talk ends
 * @param {Function} props.onToggleRecord Function to call when toggling recording
 * @param {Function} props.onModeChange Function to call when changing mode
 */
function VoiceButton({
  isRecording,
  isPushToTalk,
  isConnected,
  isMicrophoneAvailable,
  onPushToTalkStart,
  onPushToTalkEnd,
  onToggleRecord,
  onModeChange
}) {
  const [pulseAnimation, setPulseAnimation] = useState(false);
  
  // Start pulse animation when recording
  useEffect(() => {
    setPulseAnimation(isRecording);
  }, [isRecording]);
  
  // Determine button state classes
  const buttonClasses = [
    'voice-button',
    isRecording ? 'recording' : '',
    !isMicrophoneAvailable || !isConnected ? 'disabled' : '',
    pulseAnimation ? 'pulse' : ''
  ].filter(Boolean).join(' ');
  
  // Handle toggle recording
  const handleToggleClick = () => {
    if (!isConnected || !isMicrophoneAvailable) return;
    onToggleRecord();
  };
  
  // Determine button tooltip
  const getTooltip = () => {
    if (!isConnected) return 'Not connected to server';
    if (!isMicrophoneAvailable) return 'Microphone not available';
    if (isPushToTalk) return isRecording ? 'Release to send' : 'Hold to speak';
    return isRecording ? 'Tap to stop' : 'Tap to speak';
  };
  
  return (
    <div className="voice-button-container">
      {/* Main voice button */}
      {isPushToTalk ? (
        <button
          type="button"
          className={buttonClasses}
          onMouseDown={onPushToTalkStart}
          onMouseUp={onPushToTalkEnd}
          onMouseLeave={isRecording ? onPushToTalkEnd : undefined}
          onTouchStart={onPushToTalkStart}
          onTouchEnd={onPushToTalkEnd}
          onTouchCancel={isRecording ? onPushToTalkEnd : undefined}
          disabled={!isConnected || !isMicrophoneAvailable}
          title={getTooltip()}
          aria-label={getTooltip()}
        >
          <div className="button-inner">
            {isRecording ? (
              <span className="recording-icon">🎤</span>
            ) : (
              <span className="mic-icon">🎤</span>
            )}
          </div>
        </button>
      ) : (
        <button
          type="button"
          className={buttonClasses}
          onClick={handleToggleClick}
          disabled={!isConnected || !isMicrophoneAvailable}
          title={getTooltip()}
          aria-label={getTooltip()}
        >
          <div className="button-inner">
            {isRecording ? (
              <span className="recording-icon">⏹️</span>
            ) : (
              <span className="mic-icon">🎤</span>
            )}
          </div>
        </button>
      )}
      
      {/* Mode toggle */}
      <div className="mode-toggle">
        <button 
          type="button"
          className={`mode-option ${isPushToTalk ? 'active' : ''}`}
          onClick={() => onModeChange(true)}
          title="Push-to-Talk Mode"
        >
          Hold
        </button>
        <button 
          type="button"
          className={`mode-option ${!isPushToTalk ? 'active' : ''}`}
          onClick={() => onModeChange(false)}
          title="Toggle Mode"
        >
          Tap
        </button>
      </div>
      
      <div className="voice-status">
        {isRecording ? 'Listening...' : getTooltip()}
      </div>
    </div>
  );
}

export default VoiceButton;
