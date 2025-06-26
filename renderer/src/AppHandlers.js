// Extracted handlers and logic from trailing code in App.jsx
// Refactor and import these as needed in App.jsx

// Example placeholder for extracted functions
// You should refactor and import only what you need

export const handlePushToTalkEnd = async (e, {
  isRecording,
  setIsRecording,
  setMessages,
  stopRecording,
  sendMessage,
  checkMicrophoneAvailability
}) => {
  e.preventDefault();
  if (!isRecording) return;
  try {
    console.log('Stopping recording (Push-to-Talk)...');
    const audioBlob = await stopRecording();
    console.log(`Recording stopped, got audio blob of size: ${audioBlob.size} bytes`);
    if (!audioBlob || audioBlob.size < 1000) {
      console.warn('Audio recording too short or empty');
      setMessages(prev => {
        const newMessages = prev.filter(msg => msg.text !== 'Recording... Release to send.');
        return [...newMessages, {
          sender: 'system',
          text: 'Audio recording too short. Please try again and speak clearly.',
          avatar: '🎤',
          error: true,
          timestamp: Date.now()
        }];
      });
      setIsRecording(false);
      return;
    }
    const loadingMessageId = Date.now();
    setMessages(prev => {
      const newMessages = prev.filter(msg => msg.text !== 'Recording... Release to send.');
      return [...newMessages, {
        id: loadingMessageId,
        sender: 'user',
        text: 'Processing audio...',
        avatar: '👤',
        loading: true,
        timestamp: Date.now()
      }];
    });
    console.log(`Sending audio blob (${audioBlob.size} bytes) over WebSocket...`);
    const sendSuccess = sendMessage(audioBlob);
    if (!sendSuccess) {
      console.error('Failed to send audio over WebSocket');
      setMessages(prev => prev.map(msg =>
        msg.id === loadingMessageId ? {
          ...msg,
          text: 'Failed to send audio to server. Please try again.',
          loading: false,
          error: true
        } : msg
      ));
    }
  } catch (error) {
    console.error('Error stopping recording:', error);
    setMessages(prev => [...prev, {
      sender: 'system',
      text: `Error recording audio: ${error.message || 'Unknown error'}`,
      avatar: '🎤',
      error: true,
      timestamp: Date.now()
    }]);
    setIsRecording(false);
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      checkMicrophoneAvailability();
    }
  }
};

// Add more extracted logic as needed

export const handleStreamEnd = ({
  parsedMessage,
  streamingMessageId,
  messageId,
  setMessages,
  setStreamingMessageId,
  processedMessageIdsRef,
  addToAudioQueue
}) => {
  // Guard against re-processing messages that are already handled.
  if (processedMessageIdsRef.current.has(messageId)) {
    return;
  }

  const finalText = parsedMessage.full_text || '';
  const audioData = parsedMessage.audio || (parsedMessage.data && parsedMessage.data.audio);
  const audioFormat = parsedMessage.audio_format || (parsedMessage.data && parsedMessage.data.audio_format) || 'mp3';

  if (streamingMessageId === parsedMessage.id) {
    // Case 1: We were streaming this message, so update it with the final text.
    setMessages(prev => prev.map(msg => 
      msg.id === streamingMessageId 
        ? { ...msg, text: finalText, streaming: false } 
        : msg
    ));
    // Only clear the streaming ID if we've ended the correct stream.
    setStreamingMessageId(null);
  } else if (finalText) {
    // Case 2: We were not streaming, so create a new message.
    setMessages(prev => {
      if (prev.some(msg => msg.id === messageId)) return prev; // Avoid duplicates
      return [...prev, {
        id: messageId,
        sender: 'bot',
        text: finalText,
        avatar: '🤖',
        streaming: false
      }];
    });
  }

  // Always mark the message ID as processed to prevent future handling.
  processedMessageIdsRef.current.add(messageId);
  
  // Queue the audio for playback if it exists.
  if (audioData) {
    addToAudioQueue(audioData, audioFormat);
  }
};
