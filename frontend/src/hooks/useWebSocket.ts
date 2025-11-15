import { useState, useEffect, useCallback, useRef } from 'react';

interface WebSocketHookOptions {
  url: string;
  reconnectAttempts?: number;
  initialBackoffDelay?: number;
  maxBackoffDelay?: number;
  pingInterval?: number;
  onMessage?: (data: any) => void;
}

interface WebSocketHookResult {
  connected: boolean;
  error: string | null;
  sendMessage: (data: any) => void;
  reconnect: () => void;
}

export const useWebSocket = ({
  url,
  reconnectAttempts = 5,
  initialBackoffDelay = 1000,
  maxBackoffDelay = 30000,
  pingInterval = 30000,
  onMessage
}: WebSocketHookOptions): WebSocketHookResult => {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const backoffRef = useRef(initialBackoffDelay);
  const pingTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
  }, []);

  const connect = useCallback(() => {
    cleanup();
    
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        setError(null);
        attemptRef.current = 0;
        backoffRef.current = initialBackoffDelay;

        // Start ping interval
        pingTimeoutRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, pingInterval);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        cleanup();

        // Attempt reconnection with exponential backoff
        if (attemptRef.current < reconnectAttempts) {
          console.log(`Reconnecting in ${backoffRef.current}ms...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            attemptRef.current += 1;
            backoffRef.current = Math.min(
              backoffRef.current * 2,
              maxBackoffDelay
            );
            connect();
          }, backoffRef.current);
        } else {
          setError('Maximum reconnection attempts reached. Please try manually reconnecting.');
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Connection error occurred. Attempting to reconnect...');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle pong responses
          if (data.type === 'pong') {
            return;
          }

          if (onMessage) {
            onMessage(data);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
    } catch (err) {
      console.error('Error creating WebSocket:', err);
      setError('Failed to create WebSocket connection. Please check your network connection.');
    }
  }, [url, cleanup, onMessage, pingInterval, reconnectAttempts, initialBackoffDelay, maxBackoffDelay]);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    backoffRef.current = initialBackoffDelay;
    connect();
  }, [connect, initialBackoffDelay]);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      setError('Connection not open. Please wait for reconnection or try manually reconnecting.');
    }
  }, []);

  useEffect(() => {
    connect();
    return cleanup;
  }, [connect, cleanup]);

  return { connected, error, sendMessage, reconnect };
};