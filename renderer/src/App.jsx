import React, { useState, useEffect, useRef } from 'react';

import useWebSocket from './hooks/useWebSocket';
import {
  startRecording,
  stopRecording,
  checkMicrophoneAvailability,
  addToAudioQueue,
  unlockAudio
} from './utils/audio';
import './index.css';

const WEBSOCKET_URL = 'ws://127.0.0.1:8080/ws';
const MIC_AVAILABILITY_CHECK_INTERVAL = 3000;

const blobToBase64 = (blob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

import logo from '../../electron/img/logo.png';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isMicrophoneAvailable, setIsMicrophoneAvailable] = useState(true);
  const [textInput, setTextInput] = useState('');
  const [isPushToTalk, setIsPushToTalk] = useState(true);
  const [showTextInput, setShowTextInput] = useState(false);
  const [theme, setTheme] = useState('dark');
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'How can I help you today?', avatar: '🤖' },
  ]);
  const processedMessageIds = useRef(new Set());

  const messagesEndRef = useRef(null);
  const micCheckIntervalRef = useRef(null);
  const streamingMessageId = useRef(null);
  const lastProcessedIndex = useRef(-1);

  const {
    messages: wsMessages,
    isConnected,
    isPlayingAudio,
    audioQueueSize,
    stopAudio,
    sendMessage,
    sendTextMessage,
    connect,
  } = useWebSocket(WEBSOCKET_URL);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const newMessages = wsMessages.slice(lastProcessedIndex.current + 1);
    if (newMessages.length === 0) return;

    // Handle side-effects like audio playback first
    newMessages.forEach((message) => {
      if (message.type === 'tts_audio' && message.audio) {
        addToAudioQueue(message.audio, message.format, message.part_id, message.audio_id);
      }
    });

    // Now, handle state updates for the UI
    setMessages((prevMessages) => {
      let updatedMessages = [...prevMessages];

      newMessages.forEach((message) => {
        const { type, message_id, chunk, full_response, text } = message;
        const uniqueId = message_id || `msg-${Date.now()}-${type}`;

        if (processedMessageIds.current.has(uniqueId)) return;

        // We only care about UI-related messages here
        switch (type) {
          case 'stream_start':
            if (message_id && !updatedMessages.some((msg) => msg.id === message_id)) {
              updatedMessages.push({ id: message_id, sender: 'bot', text: '', avatar: '🤖', streaming: true });
            }
            streamingMessageId.current = message_id;
            break;

          case 'text_chunk':
            if (message_id && streamingMessageId.current === message_id) {
              const msgIndex = updatedMessages.findIndex((msg) => msg.id === message_id);
              if (msgIndex !== -1) {
                updatedMessages[msgIndex].text += chunk || '';
              }
            }
            break;

          case 'stream_end':
            if (message_id && streamingMessageId.current === message_id) {
              const msgIndex = updatedMessages.findIndex((msg) => msg.id === message_id);
              if (msgIndex !== -1) {
                updatedMessages[msgIndex].text = full_response || updatedMessages[msgIndex].text;
                updatedMessages[msgIndex].streaming = false;
              }
              streamingMessageId.current = null;
              processedMessageIds.current.add(uniqueId);
            }
            break;

          case 'error':
            updatedMessages.push({
              id: uniqueId,
              sender: 'system',
              text: text || 'An unknown error occurred.',
              avatar: '⚠️',
              error: true,
            });
            processedMessageIds.current.add(uniqueId);
            break;

          // No default, audio_chunk is handled outside
        }
      });

      return updatedMessages;
    });

    lastProcessedIndex.current = wsMessages.length - 1;
  }, [wsMessages]);

  useEffect(() => {
    const checkMic = async () => {
      const available = await checkMicrophoneAvailability();
      setIsMicrophoneAvailable(available);
    };
    checkMic();
    micCheckIntervalRef.current = setInterval(checkMic, MIC_AVAILABILITY_CHECK_INTERVAL);
    return () => clearInterval(micCheckIntervalRef.current);
  }, []);

  const handleManualReconnect = () => connect();
  const handleToggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');
  const handleToggleInputMode = () => setShowTextInput(!showTextInput);
  const handleToggleTalkMode = () => setIsPushToTalk(!isPushToTalk);
  const handleTextInputChange = (e) => setTextInput(e.target.value);

  const handleStopAudio = () => {
    stopAudio();
  };

  const handleSendText = () => {
    if (textInput.trim()) {
      unlockAudio();
      const userMessage = { sender: 'user', text: textInput, avatar: '👤', id: `user-${Date.now()}` };
      setMessages((prev) => [...prev, userMessage]);
      sendTextMessage(textInput);
      setTextInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSendText();
  };

  const handleVoiceButtonPress = async () => {
    if (!isMicrophoneAvailable || !isConnected) return;
    setIsRecording(true);
    try {
      await startRecording();
    } catch (error) {
      console.error('Error starting recording:', error);
      setIsRecording(false);
      setIsMicrophoneAvailable(false);
    }
  };

  const handleVoiceButtonRelease = async (e) => {
    if (!isRecording) return;
    e.preventDefault();
    setIsRecording(false);

    const userMessageId = `user-${Date.now()}`;
    try {
      const audioData = await stopRecording();
      if (audioData) {
        const placeholderMessage = { sender: 'user', text: '🎤 Audio sent', avatar: '👤', id: userMessageId };
        setMessages((prev) => [...prev, placeholderMessage]);

        const base64Audio = await blobToBase64(audioData);
        sendMessage({
          type: 'user_audio',
          audio: base64Audio.split(',')[1],
          format: audioData.type.split('/')[1].split(';')[0],
          conversation_id: 'some-conversation-id',
        });
      } else {
        console.log('No audio data recorded.');
      }
    } catch (error) {
      console.error('Error stopping recording or sending data:', error);
      setMessages((prev) =>
        prev.map((msg) => (msg.id === userMessageId ? { ...msg, text: `Error: ${error.message}`, error: true } : msg))
      );
    }
  };

  const handleVoiceButtonClick = async () => {
    if (!isMicrophoneAvailable || !isConnected) return;

    if (isRecording) {
      setIsRecording(false);
      try {
        const audioData = await stopRecording();
        if (audioData) sendMessage(audioData);
      } catch (error) {
        console.error('Error stopping recording or sending data:', error);
      }
    } else {
      setIsRecording(true);
      try {
        await startRecording();
      } catch (error) {
        console.error('Error starting recording:', error);
        setIsRecording(false);
        setIsMicrophoneAvailable(false);
      }
    }
  };

  return (
    <div className={`app-container ${theme}`}>
      <header className="app-header">
        <div className="logo" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <img src={logo} alt="Logo" style={{ height: 75, width: 75, objectFit: 'contain', borderRadius: 12 }} />
          <span style={{ fontWeight: 600, letterSpacing: '0.05em', fontSize: '1.5rem' }}>A SMART SIDEKICK</span>
        </div>
        <div className="header-icons">
          <button onClick={handleToggleTheme} className="icon-button">{theme === 'dark' ? '☀️' : '🌙'}</button>
          <button onClick={handleToggleInputMode} className="icon-button">⌨️</button>
        </div>
      </header>

      <main className="chat-container">
        {messages.map((msg, index) => (
          <div key={msg.id || index} className={`message ${msg.sender}`}>
            <div className="avatar">{msg.avatar}</div>
            <div className="message-bubble">
              {msg.text}
              {msg.streaming && <span className="streaming-indicator"></span>}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </main>

      <footer className="app-footer">
        {(isPlayingAudio || audioQueueSize > 0) && (
          <div className="playback-status-container">
            <div className="playback-info">
              <span className="wave-icon">🔊</span>
              <span>{isPlayingAudio ? 'Speaking...' : 'Buffering...'}</span>
              {audioQueueSize > 0 && <span className="queue-size">({audioQueueSize})</span>}
            </div>
            <button onClick={handleStopAudio} className="stop-button">Stop</button>
          </div>
        )}

        <div className="status-bar">
          <span>Status: {isConnected ? 'Connected' : 'Disconnected'}</span>
          {!isMicrophoneAvailable && <span className="mic-status">🎤 Error</span>}
          {!isConnected && <button onClick={handleManualReconnect} className="reconnect-button">Connect</button>}
        </div>

        {showTextInput ? (
          <div className="text-input-container">
            <input
              type="text"
              value={textInput}
              onChange={handleTextInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              disabled={!isConnected}
            />
            <button onClick={handleSendText} disabled={!textInput.trim() || !isConnected}>Send</button>
          </div>
        ) : (
          <div className="voice-controls">
            <button
              id="voice-button"
              className={isRecording ? 'recording' : ''}
              onMouseDown={isPushToTalk ? handleVoiceButtonPress : undefined}
              onMouseUp={isPushToTalk ? handleVoiceButtonRelease : undefined}
              onClick={!isPushToTalk ? handleVoiceButtonClick : undefined}
              disabled={!isMicrophoneAvailable || !isConnected}
            >
              {isRecording ? '...' : '🎤'}
            </button>
            <div className="talk-mode-toggle">
              <label>
                <input type="checkbox" checked={isPushToTalk} onChange={handleToggleTalkMode} />
                Push-to-Talk
              </label>
            </div>
          </div>
        )}
      </footer>
    </div>
  );
}

export default App;