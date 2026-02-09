/**
 * Hook for connecting to live match WebSocket stream
 *
 * Uses a frame buffer to smooth playback over network jitter.
 * Frames are buffered until we have enough, then played back at a steady rate.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  ArenaMessage,
  MatchStartMessage,
  PlayerFrameData,
} from "../lib/liveMatchProtocol";
import type { MatchFrame, PlayerFrame } from "../components/viewer/MeleeViewer";
// Character IDs from arena are already internal IDs (libmelee format)
import { ARENA_URL } from "../config";

// Buffer config: wait for this many frames before starting playback
// WebSocket delivers frames continuously at 60fps, so smaller buffer is fine
const BUFFER_TARGET = 8;  // ~133ms latency
const PLAYBACK_INTERVAL_MS = 16; // ~60fps playback
const MAX_BUFFER_SIZE = 30; // Drop frames if we fall this far behind

export interface LiveMatchState {
  status: "connecting" | "connected" | "ended" | "error" | "disconnected";
  matchInfo: MatchStartMessage | null;
  currentFrame: MatchFrame | null;
  error: string | null;
  gameScore: [number, number];
  bufferHealth: number; // 0-1, how full the buffer is relative to target
}

interface UseLiveMatchOptions {
  onMatchStart?: (info: MatchStartMessage) => void;
  onMatchEnd?: () => void;
  onError?: (error: string) => void;
}

/**
 * Connect to a live match stream
 *
 * @param matchId - The match ID to connect to (or null to disconnect)
 * @param options - Callbacks for match events
 */
export function useLiveMatch(
  matchId: string | null,
  options: UseLiveMatchOptions = {}
): LiveMatchState {
  const [state, setState] = useState<LiveMatchState>({
    status: "disconnected",
    matchInfo: null,
    currentFrame: null,
    error: null,
    gameScore: [0, 0],
    bufferHealth: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Frame buffer for smooth playback
  const frameBufferRef = useRef<MatchFrame[]>([]);
  const playbackIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bufferingRef = useRef(true); // Start in buffering mode
  const matchInfoRef = useRef<MatchStartMessage | null>(null); // For frame handler access

  // Convert arena PlayerFrameData to viewer PlayerFrame
  const convertPlayerFrame = useCallback(
    (data: PlayerFrameData, matchInfo: MatchStartMessage): PlayerFrame => {
      // Find player info from match start
      const playerInfo = matchInfo.players.find((p) => p.port === data.port);
      // characterId from arena is libmelee's internal ID (same as animation files use)
      const internalCharId = playerInfo?.characterId ?? 0;

      return {
        internalCharId,
        x: data.x,
        y: data.y,
        actionStateId: data.actionStateId,
        actionFrame: data.actionFrame,
        facingDirection: data.facingDirection,
        percent: data.percent,
        stocks: data.stocks,
      };
    },
    []
  );

  // Start smooth playback from buffer
  const startPlayback = useCallback(() => {
    if (playbackIntervalRef.current) return; // Already playing

    console.log(`[useLiveMatch] Starting playback, buffer has ${frameBufferRef.current.length} frames`);

    let lastFrameTime = performance.now();
    const targetFrameTime = PLAYBACK_INTERVAL_MS;

    const tick = (now: number) => {
      if (!playbackIntervalRef.current) return; // Stopped

      const elapsed = now - lastFrameTime;

      // Only advance frame if enough time has passed
      if (elapsed >= targetFrameTime) {
        lastFrameTime = now - (elapsed % targetFrameTime); // Account for drift

        // If buffer is getting too large, we're falling behind - drop old frames
        if (frameBufferRef.current.length > MAX_BUFFER_SIZE) {
          const toDrop = frameBufferRef.current.length - BUFFER_TARGET;
          frameBufferRef.current.splice(0, toDrop);
          console.log(`[useLiveMatch] Dropped ${toDrop} frames to catch up`);
        }

        // Simple FIFO - frames arrive in order from WebSocket
        const frame = frameBufferRef.current.shift();

        if (frame) {
          setState((prev) => ({
            ...prev,
            currentFrame: frame,
            bufferHealth: Math.min(1, frameBufferRef.current.length / BUFFER_TARGET),
          }));
        }
      }

      requestAnimationFrame(tick);
    };

    // Use a dummy interval ref to track "playing" state, actual timing via rAF
    playbackIntervalRef.current = 1 as unknown as ReturnType<typeof setInterval>;
    requestAnimationFrame(tick);
  }, []);

  // Stop playback
  const stopPlayback = useCallback(() => {
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
    }
    frameBufferRef.current = [];
    bufferingRef.current = true;
    matchInfoRef.current = null;
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const msg: ArenaMessage = JSON.parse(event.data);

        // Debug logging for all messages
        if (msg.type === "frame") {
          // Log first frame and every 60th after
          const frameNum = (msg as { frame: number }).frame;
          if (frameNum === 0 || frameNum % 60 === 0) {
            console.log(`[useLiveMatch] Frame ${frameNum}, buffer: ${frameBufferRef.current.length}, buffering: ${bufferingRef.current}`);
          }
        } else {
          console.log(`[useLiveMatch] Received message:`, msg.type, msg);
        }

        switch (msg.type) {
          case "match_start":
            console.log(`[useLiveMatch] match_start received:`, msg);
            // Clear the connect timeout since we got match info
            if (wsRef.current) {
              const ws = wsRef.current as WebSocket & { _connectTimeout?: ReturnType<typeof setTimeout> };
              if (ws._connectTimeout) {
                clearTimeout(ws._connectTimeout);
                ws._connectTimeout = undefined;
              }
            }
            matchInfoRef.current = msg; // Store in ref for frame handler
            setState((prev) => ({
              ...prev,
              status: "connected",
              matchInfo: msg,
              error: null,
            }));
            optionsRef.current.onMatchStart?.(msg);
            break;

          case "frame":
            // Add frame to buffer (don't setState here - too expensive at 60fps)
            {
              const matchInfo = matchInfoRef.current;
              if (!matchInfo) {
                console.log(`[useLiveMatch] Frame received but no matchInfo yet, ignoring`);
                break;
              }

              const players = msg.players.map((p) =>
                convertPlayerFrame(p, matchInfo)
              );

              const newFrame: MatchFrame = {
                frame: msg.frame,
                stageId: matchInfo.stageId,
                players,
              };

              // Add to buffer (just update ref, no re-render)
              frameBufferRef.current.push(newFrame);

              // Start playback once buffer is full enough
              if (bufferingRef.current && frameBufferRef.current.length >= BUFFER_TARGET) {
                bufferingRef.current = false;
                startPlayback();
              }
            }
            break;

          case "game_end":
            setState((prev) => {
              const newScore: [number, number] = [...prev.gameScore];
              // Assuming 2-player match, port 1 is index 0, port 2 is index 1
              if (msg.winnerPort === 1) newScore[0]++;
              else newScore[1]++;
              return { ...prev, gameScore: newScore };
            });
            break;

          case "match_end":
            setState((prev) => ({
              ...prev,
              status: "ended",
              gameScore: msg.finalScore,
            }));
            optionsRef.current.onMatchEnd?.();
            break;

          case "error":
            setState((prev) => ({
              ...prev,
              status: "error",
              error: msg.message,
            }));
            optionsRef.current.onError?.(msg.message);
            break;
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    },
    [convertPlayerFrame, startPlayback]
  );

  // Connect/disconnect based on matchId
  useEffect(() => {
    if (!matchId) {
      // Disconnect
      stopPlayback();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setState({
        status: "disconnected",
        matchInfo: null,
        currentFrame: null,
        error: null,
        gameScore: [0, 0],
        bufferHealth: 0,
      });
      return;
    }

    // Build WebSocket URL
    const wsUrl = ARENA_URL.replace(/^http/, "ws") + `/ws/match/${matchId}`;

    setState((prev) => ({ ...prev, status: "connecting", error: null }));

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`[useLiveMatch] Connected to ${wsUrl}`);

      // Set a timeout - if we don't receive match_start within 5s, the match may have ended
      const timeout = setTimeout(() => {
        setState((prev) => {
          if (prev.status === "connecting") {
            console.log("[useLiveMatch] Timeout waiting for match_start - match may have ended");
            return {
              ...prev,
              status: "ended",
              error: "Match has already ended",
            };
          }
          return prev;
        });
      }, 5000);

      // Store timeout so we can clear it if we do receive match_start
      (ws as WebSocket & { _connectTimeout?: ReturnType<typeof setTimeout> })._connectTimeout = timeout;
    };

    ws.onmessage = handleMessage;

    ws.onerror = (event) => {
      console.error("[useLiveMatch] WebSocket error:", event);
      setState((prev) => ({
        ...prev,
        status: "error",
        error: "Connection error",
      }));
    };

    ws.onclose = (event) => {
      console.log(`[useLiveMatch] Disconnected: ${event.code} ${event.reason}`);
      if (wsRef.current === ws) {
        setState((prev) => ({
          ...prev,
          status: prev.status === "ended" ? "ended" : "disconnected",
        }));
      }
    };

    return () => {
      // Clear any pending timeout
      const wsWithTimeout = ws as WebSocket & { _connectTimeout?: ReturnType<typeof setTimeout> };
      if (wsWithTimeout._connectTimeout) {
        clearTimeout(wsWithTimeout._connectTimeout);
      }
      // Stop playback
      stopPlayback();
      ws.close();
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, [matchId, handleMessage, stopPlayback]);

  return state;
}
