import { useState, useEffect, useRef, useCallback } from 'react';

// Track instances of the hook to detect multiple mounts
let hookInstanceCounter = 0;

const useWebSocket = (providedUrl) => {
  // Add unique instance ID to detect multiple hook instances
  const hookInstanceId = useRef(hookInstanceCounter++);
  
  // Use relative WebSocket URL to avoid hostname mismatches
  const url = providedUrl.startsWith('ws://') ? 
    providedUrl : 
    `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
  
  // Reduce log noise during startup
  const isInitialMount = useRef(true);
  const isInitialConnection = useRef(true);
  const hasConnectedBefore = useRef(false);
  
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('initializing');
  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pingInterval = useRef(null);
  const MAX_RECONNECT_DELAY = 5000; // Maximum reconnect delay in ms
  const MAX_RECONNECT_ATTEMPTS = 10; // Maximum number of reconnect attempts

  // Clear any existing reconnect timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  // Send a ping message through the WebSocket
  const sendPing = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      const ping = { 
        type: "ping", 
        timestamp: Date.now(),
        clientId: `interval-${Date.now()}`,
        client: "react-frontend"
      };
      
      try {
        ws.current.send(JSON.stringify(ping));
        return true;
      } catch (error) {
        console.error('Error sending ping:', error);
        return false;
      }
    }
    return false;
  }, []);

  // Start ping interval for keeping connection alive
  const startPingInterval = useCallback(() => {
    // Clear any existing ping interval
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
    }
    
    // Start a new ping interval
    pingInterval.current = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        sendPing();
      } else if (ws.current?.readyState === WebSocket.CLOSED || ws.current?.readyState === WebSocket.CLOSING) {
        if (!isInitialConnection.current) {
          console.warn('WebSocket closed or closing in ping interval');
        }
      }
    }, 3000); // Send ping every 3 seconds for better stability
  }, [sendPing]);

  // Create WebSocket connection
  const createWebSocket = useCallback(() => {
    // Don't try to reconnect if we've reached the maximum number of attempts
    if (connectionAttempts >= MAX_RECONNECT_ATTEMPTS) {
      setConnectionStatus('failed');
      return;
    }
    
    // Clear any existing WebSocket
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    
    // Clear any existing reconnect timeout
    clearReconnectTimeout();
    
    // Clear any existing ping interval
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
      pingInterval.current = null;
    }
    
    try {
      // Only log connection attempts after the first one to reduce noise
      if (!isInitialConnection.current || connectionAttempts > 0) {
        setConnectionStatus('connecting');
        console.log(`Connecting to WebSocket (attempt ${connectionAttempts + 1})...`);
      } else {
        console.log('Establishing initial WebSocket connection...');
      }
      
      const socket = new WebSocket(url);
      ws.current = socket;
      
      socket.onopen = () => {
        hasConnectedBefore.current = true;
        isInitialConnection.current = false;
        
        console.log('WebSocket connected!');
        setIsConnected(true);
        setConnectionStatus('connected');
        setConnectionAttempts(0); // Reset connection attempts on successful connection
        
        // Send initial ping to establish connection
        const initialPing = { 
          type: "ping", 
          timestamp: Date.now(),
          clientId: `${Date.now()}-${Math.random().toString(36).substring(2, 8)}`,
          client: "react-frontend"
        };
        
        try {
          socket.send(JSON.stringify(initialPing));
          console.log('Initial ping sent');
        } catch (error) {
          console.error('Error sending initial ping:', error);
        }
        
        // Start ping interval
        startPingInterval();
      };

      socket.onmessage = (event) => {
        try {
          let parsedMessage;
          
          // Try to parse as JSON
          try {
            parsedMessage = JSON.parse(event.data);
          } catch (e) {
            // If not JSON, use the raw data
            parsedMessage = event.data;
          }
          
          // Add the message to state
          setMessages(prev => [...prev, parsedMessage]);
        } catch (error) {
          console.error('Error handling WebSocket message:', error);
        }
      };

      socket.onclose = (event) => {
        if (hasConnectedBefore.current) {
          console.log(`WebSocket closed with code ${event.code}, reason: ${event.reason}`);
          setIsConnected(false);
          setConnectionStatus('disconnected');
        }
        
        // Attempt to reconnect with exponential backoff
        const delay = Math.min(1000 * Math.pow(1.5, connectionAttempts), MAX_RECONNECT_DELAY);
        
        if (connectionAttempts < MAX_RECONNECT_ATTEMPTS) {
          if (hasConnectedBefore.current) {
            console.log(`Reconnecting in ${delay}ms (attempt ${connectionAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})...`);
          }
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setConnectionAttempts(prev => prev + 1);
            createWebSocket();
          }, delay);
        } else {
          console.error(`Maximum reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached.`);
          setConnectionStatus('failed');
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      
      // Attempt to reconnect with exponential backoff
      const delay = Math.min(1000 * Math.pow(1.5, connectionAttempts), MAX_RECONNECT_DELAY);
      
      if (connectionAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectTimeoutRef.current = setTimeout(() => {
          setConnectionAttempts(prev => prev + 1);
          createWebSocket();
        }, delay);
      } else {
        console.error(`Maximum reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached.`);
        setConnectionStatus('failed');
      }
    }
  }, [url, connectionAttempts, clearReconnectTimeout, startPingInterval, MAX_RECONNECT_ATTEMPTS, MAX_RECONNECT_DELAY]);

  // Connect to the WebSocket server
  const connect = useCallback(() => {
    // Reset connection attempts when manually connecting
    setConnectionAttempts(0);
    createWebSocket();
  }, [createWebSocket]);

  // Disconnect from the WebSocket server
  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close(1000, 'User initiated disconnect');
    }
    
    clearReconnectTimeout();
    
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
      pingInterval.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, [clearReconnectTimeout]);

  // Generic message sender
  const sendMessage = useCallback((message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        // Check if message is a Blob or ArrayBuffer (binary data)
        if (message instanceof Blob || message instanceof ArrayBuffer) {
          console.log(`📤 Sending binary data: ${message instanceof Blob ? 'Blob' : 'ArrayBuffer'} of size ${message instanceof Blob ? message.size : message.byteLength} bytes`);
          ws.current.send(message);
          return true;
        }
        
        // For non-binary data, ensure message is a string
        const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
        console.log(`📤 Sending text message: ${messageStr.substring(0, 100)}${messageStr.length > 100 ? '...' : ''}`);
        ws.current.send(messageStr);
        return true;
      } catch (error) {
        console.error('❌ Error sending message:', error);
        return false;
      }
    } else {
      console.warn('⚠️ Cannot send message, WebSocket is not connected');
      return false;
    }
  }, []);

  // Text message sender (convenience method)
  const sendTextMessage = useCallback((text) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        const message = {
          type: "text",
          text: text,
          timestamp: Date.now()
        };
        ws.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Error sending text message:', error);
        return false;
      }
    } else {
      console.warn('Cannot send text message, WebSocket is not connected');
      return false;
    }
  }, []);

  // Initialize WebSocket connection on mount
  useEffect(() => {
    // Add a small delay before initial connection to reduce startup noise
    const initialDelay = isInitialMount.current ? 500 : 0;
    
    const timer = setTimeout(() => {
      createWebSocket();
      isInitialMount.current = false;
    }, initialDelay);
    
    return () => {
      clearTimeout(timer);
      
      // Clean up WebSocket on unmount
      if (ws.current) {
        ws.current.close(1000, 'Component unmounting');
      }
      
      clearReconnectTimeout();
      
      if (pingInterval.current) {
        clearInterval(pingInterval.current);
        pingInterval.current = null;
      }
    };
  }, [createWebSocket, clearReconnectTimeout]);

  return {
    messages,
    isConnected,
    connectionStatus,
    connectionAttempts,
    maxReconnectAttempts: MAX_RECONNECT_ATTEMPTS,
    connect,
    disconnect,
    sendMessage,
    sendTextMessage,
    sendPing
  };
};

export default useWebSocket;
