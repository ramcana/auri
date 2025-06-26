import { useState, useEffect, useRef, useCallback } from 'react';
import {
  stopAudioPlayback,
  clearAudioQueue,
  getAudioPlaybackState,
  getAudioQueue,
} from '../utils/audio';

// Track instances of the hook to detect multiple mounts
let hookInstanceCounter = 0;

const useWebSocket = (providedUrl) => {

  const url = providedUrl.startsWith('ws://')
    ? providedUrl
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [playbackState, setPlaybackState] = useState({ isPlaying: false, queueSize: 0 });

  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pingInterval = useRef(null);
  const audioStateInterval = useRef(null);
  const isUnmounting = useRef(false);
  const messageBuffer = useRef([]);
  const messageTimer = useRef(null);

  const MAX_RECONNECT_DELAY = 5000;
  const MAX_RECONNECT_ATTEMPTS = 10;

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const sendPing = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      return true;
    }
    return false;
  }, []);

  const startPingInterval = useCallback(() => {
    if (pingInterval.current) clearInterval(pingInterval.current);
    pingInterval.current = setInterval(() => {
      if (ws.current?.readyState !== WebSocket.OPEN) {
        console.warn('Cannot send ping, WebSocket is not open.');
      }
      sendPing();
    }, 3000);
  }, [sendPing]);

  const createWebSocket = useCallback(() => {
    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log('🟢 WebSocket connection opened');
        setIsConnected(true);
        setConnectionAttempts(0);
        sendPing();
        startPingInterval();
      };

      ws.current.onmessage = (event) => {
        try {
          const parsedMessage = JSON.parse(event.data);
          if (parsedMessage.type !== 'pong') {
            messageBuffer.current.push(parsedMessage);

            if (!messageTimer.current) {
              messageTimer.current = setTimeout(() => {
                setMessages((prev) => [...prev, ...messageBuffer.current]);
                messageBuffer.current = [];
                messageTimer.current = null;
              }, 50); // Batch messages
            }
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.current.onclose = (event) => {
        console.warn(`🔴 WebSocket connection closed: ${event.code} ${event.reason}`);
        setIsConnected(false);
        if (pingInterval.current) clearInterval(pingInterval.current);

        if (isUnmounting.current || event.code === 1000) return;

        if (connectionAttempts < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(1000 * 2 ** connectionAttempts, MAX_RECONNECT_DELAY);
          console.log(`Retrying connection in ${delay}ms... (Attempt ${connectionAttempts + 1})`);
          reconnectTimeoutRef.current = setTimeout(() => {
            setConnectionAttempts((prev) => prev + 1);
            connect();
          }, delay);
        } else {
          console.error('Max reconnection attempts reached.');
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
    }
  }, [url, startPingInterval, connectionAttempts]);

  const connect = useCallback(() => {
    if (ws.current && ws.current.readyState !== WebSocket.CLOSED) return;
    clearReconnectTimeout();
    createWebSocket();
  }, [createWebSocket, clearReconnectTimeout]);

  useEffect(() => {
    if (!url) return;
    connect();

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && ws.current?.readyState !== WebSocket.OPEN) {
        connect();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Poll for audio state changes
    audioStateInterval.current = setInterval(() => {
      const state = getAudioPlaybackState();
      const queue = getAudioQueue();
      setPlaybackState(prevState => {
        if (prevState.isPlaying !== state.isPlaying || prevState.queueSize !== queue.length) {
          return { isPlaying: state.isPlaying, queueSize: queue.length };
        }
        return prevState;
      });
    }, 200);

    return () => {
      isUnmounting.current = true;
      clearReconnectTimeout();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (pingInterval.current) clearInterval(pingInterval.current);
      if (audioStateInterval.current) clearInterval(audioStateInterval.current);
      if (messageTimer.current) clearTimeout(messageTimer.current);
      if (ws.current) {
        ws.current.close(1000, 'Component unmounting');
      }
    };
  }, [url, connect, clearReconnectTimeout]);

  const sendTextMessage = useCallback((text) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'text', text, timestamp: Date.now() }));
      return true;
    }
    return false;
  }, []);

  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      try {
        const data = typeof message === 'string' || message instanceof Blob || message instanceof ArrayBuffer
          ? message
          : JSON.stringify(message);
        ws.current.send(data);
        return true;
      } catch (error) {
        console.error('Error sending message:', error);
        return false;
      }
    }
    return false;
  }, []);

  const stopAudio = useCallback(() => {
    stopAudioPlayback();
    clearAudioQueue();
  }, []);

  return {
    messages,
    isConnected,
    isPlayingAudio: playbackState.isPlaying,
    audioQueueSize: playbackState.queueSize,
    stopAudio,
    sendMessage,
    sendTextMessage,
    connect,
  };
};

export default useWebSocket;
