import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import MessageList from './components/MessageList';
import { startRecording, stopRecording, checkMicrophoneAvailability, playAudioFromBase64 } from './utils/audio';
import useWebSocket from './hooks/useWebSocket';

// Placeholder icons (replace with actual SVGs or an icon library later)
const SettingsIcon = () => <span className="icon">⚙️</span>; // Gear emoji
const UserIcon = () => <span className="icon">👤</span>; // Person silhouette emoji
const MicrophoneIcon = () => <span className="icon">🎤</span>;
const MicrophoneOffIcon = () => <span className="icon">🔇</span>; // Muted microphone emoji
const SendIcon = () => <span className="icon">➤</span>; // Right arrow emoji

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isMicrophoneAvailable, setIsMicrophoneAvailable] = useState(true); // Optimistically assume microphone is available
  const [textInput, setTextInput] = useState('');
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'How can I help you today?', avatar: '🤖' }, // Robot emoji for bot
  ]);
  
  // Define WebSocket URL directly to avoid any potential undefined issues
  const wsUrl = "ws://127.0.0.1:8080/ws";
  
  // Track connection attempts for UI feedback
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  
  // Use the WebSocket hook directly without useMemo to avoid potential issues
  const { 
    messages: wsMessages, 
    isConnected, 
    connectionStatus,
    sendMessage, 
    sendTextMessage, 
    connect,
    maxReconnectAttempts 
  } = useWebSocket(wsUrl);
  
  // Manual reconnect function with visual feedback
  const handleManualReconnect = () => {
    setMessages(prev => [...prev, { 
      sender: 'system', 
      text: 'Manually reconnecting to server...', 
      avatar: '🔄',
      timestamp: Date.now()
    }]);
    connect(); // Call the connect function from the WebSocket hook
  };
  
  // Track if we've shown the initial connection message
  const initialConnectionMessageShown = useRef(false);
  const lastConnectionStatus = useRef(null);
  
  // Update messages when WebSocket connection status changes
  useEffect(() => {
    // Get the connection status from the WebSocket hook
    const status = connectionStatus || (isConnected ? 'connected' : 'disconnected');
    
    // Skip status updates during initial startup to reduce noise
    if (status === 'initializing') {
      return;
    }
    
    // Prevent duplicate messages for the same status
    if (lastConnectionStatus.current === status) {
      return;
    }
    
    // Update the last status
    lastConnectionStatus.current = status;
    
    // Handle different connection statuses
    switch (status) {
      case 'connected':
        // Reset connection attempts when successfully connected
        setConnectionAttempts(0);
        
        // Only show the connection message once
        if (!initialConnectionMessageShown.current) {
          initialConnectionMessageShown.current = true;
          setMessages(prev => [...prev, { 
            sender: 'system', 
            text: 'Connected to voice bot server', 
            avatar: '🔌',
            timestamp: Date.now()
          }]);
        }
        break;
        
      case 'failed':
        setMessages(prev => {
          // Only add max reconnection message if it's not already the last message
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.text === 'Disconnected from server. Max reconnection attempts reached. Please try manually reconnecting.') {
            return prev;
          }
          return [...prev, { 
            sender: 'system', 
            text: 'Disconnected from server. Max reconnection attempts reached. Please try manually reconnecting.', 
            avatar: '🔌',
            timestamp: Date.now()
          }];
        });
        break;
        
      case 'disconnected':
        // Only show disconnection messages if we've successfully connected before
        if (initialConnectionMessageShown.current) {
          setMessages(prev => {
            // Only add reconnecting message if it's not already the last message
            const lastMsg = prev[prev.length - 1];
            if (lastMsg && lastMsg.text === 'Disconnected from server. Attempting to reconnect...') {
              return prev;
            }
            return [...prev, { 
              sender: 'system', 
              text: 'Disconnected from server. Attempting to reconnect...', 
              avatar: '🔌',
              timestamp: Date.now()
            }];
          });
        }
        break;
        
      default:
        // No message for other states
        break;
    }
  }, [connectionStatus, isConnected, connectionAttempts]);

  const textInputRef = useRef(null);
  
  // Check microphone availability when component mounts
  useEffect(() => {
    const checkMicrophone = async () => {
      const available = await checkMicrophoneAvailability();
      setIsMicrophoneAvailable(available);
      
      if (!available) {
        setMessages(prev => [
          ...prev, 
          { 
            sender: 'system', 
            text: 'No microphone detected. You can use text input instead.', 
            avatar: '🎤',
            timestamp: Date.now()
          }
        ]);
      }
    };
    
    checkMicrophone();
  }, []);
  
  // Process WebSocket messages
  useEffect(() => {
    if (!wsMessages || wsMessages.length === 0) return;
    
    // Get the last message
    const lastMessage = wsMessages[wsMessages.length - 1];
    
    try {
      // Try to parse the message if it's a string
      const parsedMessage = typeof lastMessage === 'string' ? JSON.parse(lastMessage) : lastMessage;
      
      // Handle bot responses
      if (parsedMessage.type === 'bot_response') {
        setMessages(prev => {
          // Find and remove any loading messages
          const newMessages = prev.filter(m => !m.loading);
          
          // Add the bot response
          return [
            ...newMessages, 
            { 
              id: `bot-${Date.now()}`,
              sender: 'bot', 
              text: parsedMessage.text, 
              avatar: '🤖',
              hasAudio: !!parsedMessage.audio,
              audio: parsedMessage.audio,
              audio_format: parsedMessage.audio_format,
              duration_ms: parsedMessage.duration_ms,
              timestamp: Date.now()
            }
          ];
        });
        
        // Auto-play the audio if available
        if (parsedMessage.audio) {
          console.log(`Auto-playing audio response (${parsedMessage.audio_format})...`);
          playAudioFromBase64(parsedMessage.audio, parsedMessage.audio_format)
            .catch(error => {
              console.error('Error playing audio response:', error);
            });
        }
      } 
      // Handle transcription results
      else if (parsedMessage.type === 'transcription_result') {
        setMessages(prev => {
          // Replace loading message with transcription
          return prev.map(msg => {
            if (msg.loading && msg.sender === 'user') {
              return {
                ...msg,
                text: parsedMessage.text,
                loading: false,
                timestamp: Date.now()
              };
            }
            return msg;
          });
        });
      }
      // Handle error messages
      else if (parsedMessage.type === 'error' || parsedMessage.type === 'transcription_error') {
        setMessages(prev => {
          // Replace loading message with error
          return prev.map(msg => {
            if (msg.loading) {
              return {
                ...msg,
                text: parsedMessage.text || 'Error processing audio',
                loading: false,
                error: true,
                timestamp: Date.now()
              };
            }
            return msg;
          });
        });
      }
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
    }
  }, [wsMessages]);

  // Handle record button click
  const handleRecord = async () => {
    if (!isConnected) {
      console.warn('Cannot record audio: WebSocket not connected');
      setMessages(prev => [...prev, { 
        sender: 'system', 
        text: 'Cannot record audio: Not connected to server', 
        avatar: '🎤',
        error: true,
        timestamp: Date.now()
      }]);
      return;
    }

    if (isRecording) {
      try {
        console.log('Stopping recording...');
        const audioBlob = await stopRecording();
        console.log(`Recording stopped, got audio blob of size: ${audioBlob.size} bytes`);
        
        // Validate audio blob size
        if (!audioBlob || audioBlob.size < 1000) { // Less than 1KB is probably too small
          console.warn('Audio recording too short or empty');
          setMessages(prev => [...prev, { 
            sender: 'system', 
            text: 'Audio recording too short. Please try again and speak clearly.', 
            avatar: '🎤',
            error: true,
            timestamp: Date.now()
          }]);
          setIsRecording(false);
          return;
        }
        
        // Add a loading message
        const loadingMessageId = Date.now();
        setMessages(prev => [...prev, { 
          id: loadingMessageId,
          sender: 'user', 
          text: 'Processing audio...', 
          avatar: '👤',
          loading: true,
          timestamp: Date.now()
        }]);
        
        // Send the audio blob directly over WebSocket
        console.log(`Sending audio blob (${audioBlob.size} bytes) over WebSocket...`);
        const sendSuccess = sendMessage(audioBlob);
        
        if (!sendSuccess) {
          console.error('Failed to send audio over WebSocket');
          // Replace loading message with error
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
      } finally {
        setIsRecording(false);
      }
    } else {
      try {
        // Check if microphone is available before starting
        if (!isMicrophoneAvailable) {
          throw new Error('Microphone not available or permission denied');
        }
        
        console.log('Starting recording...');
        await startRecording();
        console.log('Recording started');
        setIsRecording(true);
        
        // Add a recording indicator message
        setMessages(prev => [...prev, { 
          sender: 'system', 
          text: 'Recording... Speak now and click the microphone button when finished.', 
          avatar: '🎤',
          timestamp: Date.now()
        }]);
      } catch (error) {
        console.error('Error starting recording:', error);
        setMessages(prev => [...prev, { 
          sender: 'system', 
          text: `Error starting microphone: ${error.message || 'Permission denied'}`, 
          avatar: '🎤',
          error: true,
          timestamp: Date.now()
        }]);
        
        // Request microphone permission again if it was denied
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
          checkMicrophoneAvailability();
        }
      }
    }
  };

  // Handle text input submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!textInput.trim()) return;
    
    // Add user message
    setMessages(prev => [...prev, { 
      sender: 'user', 
      text: textInput, 
      avatar: '👤',
      timestamp: Date.now()
    }]);
    
    // Send message to server
    if (isConnected) {
      sendTextMessage(textInput);
    } else {
      setMessages(prev => [...prev, { 
        sender: 'system', 
        text: 'Cannot send message: Not connected to server', 
        avatar: '🔌',
        error: true,
        timestamp: Date.now()
      }]);
    }
    
    // Clear input
    setTextInput('');
  };

  // Focus text input when component mounts
  useEffect(() => {
    if (textInputRef.current) {
      textInputRef.current.focus();
    }
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Voice Bot</h1>
        <div className="connection-status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></span>
          <span className="status-text">{isConnected ? 'Connected' : 'Disconnected'}</span>
          {!isConnected && (
            <button className="reconnect-button" onClick={handleManualReconnect}>
              Reconnect
            </button>
          )}
        </div>
        <button className="settings-button">
          <SettingsIcon />
        </button>
      </header>
      
      <main className="app-main">
        <MessageList messages={messages} />
      </main>
      
      <footer className="app-footer">
        <form className="input-form" onSubmit={handleSubmit}>
          <button
            type="button"
            className={`record-button ${isRecording ? 'recording' : ''} ${!isMicrophoneAvailable ? 'disabled' : ''}`}
            onClick={handleRecord}
            disabled={!isMicrophoneAvailable}
          >
            {isRecording ? <MicrophoneIcon /> : (isMicrophoneAvailable ? <MicrophoneIcon /> : <MicrophoneOffIcon />)}
          </button>
          
          <input
            type="text"
            className="text-input"
            placeholder="Type a message..."
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            ref={textInputRef}
          />
          
          <button type="submit" className="send-button" disabled={!textInput.trim()}>
            <SendIcon />
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
