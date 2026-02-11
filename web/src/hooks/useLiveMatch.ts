/**
 * Hook for connecting to live match WebSocket stream
 *
 * Uses a circular frame buffer to smooth playback over network jitter.
 * Frames are buffered until we have enough, then played back at a steady rate.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  ArenaMessage,
  MatchStartMessage,
  PlayerFrameData,
} from "../lib/liveMatchProtocol";
import type { MatchFrame, PlayerFrame } from "../components/viewer/MeleeViewer";
import { ARENA_URL } from "../config";

// Buffer config — tuned for Slippi rollback netcode
// Slippi uses deterministic rollback with online_delay frames (typically 6).
// During rollback, libmelee re-emits frames with duplicate/backward frame numbers.
// Buffer must be larger than rollback window + network jitter to absorb this.
const BUFFER_TARGET = 20; // ~333ms latency — absorbs 6-frame rollback + 6-frame delay + jitter
const PLAYBACK_INTERVAL_MS = 16; // ~60fps playback
const BUFFER_CAPACITY = 128; // Circular buffer size (power of 2 for fast modulo)
const CATCHUP_RATE = 4; // Extra frames to consume per tick when buffer is overfull
const OVERFLOW_THRESHOLD = BUFFER_TARGET * 3; // Start catching up above this

// ---------------------------------------------------------------------------
// O(1) circular buffer — replaces Array.shift()/splice() which are O(n)
// ---------------------------------------------------------------------------

interface RingBuffer<T> {
  items: (T | undefined)[];
  read: number;
  write: number;
  size: number;
}

function createRing<T>(capacity: number): RingBuffer<T> {
  return { items: new Array(capacity), read: 0, write: 0, size: 0 };
}

function ringPush<T>(buf: RingBuffer<T>, item: T): void {
  const cap = buf.items.length;
  buf.items[buf.write] = item;
  buf.write = (buf.write + 1) % cap;
  if (buf.size < cap) {
    buf.size++;
  } else {
    // Overwrite oldest — advance read pointer
    buf.read = (buf.read + 1) % cap;
  }
}

function ringShift<T>(buf: RingBuffer<T>): T | undefined {
  if (buf.size === 0) return undefined;
  const item = buf.items[buf.read];
  buf.items[buf.read] = undefined; // help GC
  buf.read = (buf.read + 1) % buf.items.length;
  buf.size--;
  return item;
}

// ---------------------------------------------------------------------------

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
  options: UseLiveMatchOptions = {},
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

  // Reconnection state
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const endedRef = useRef(false);

  // Circular frame buffer for smooth playback
  const frameBufferRef = useRef<RingBuffer<MatchFrame>>(createRing(BUFFER_CAPACITY));
  const playingRef = useRef(false);
  const bufferingRef = useRef(true);
  const matchInfoRef = useRef<MatchStartMessage | null>(null);

  // Frame deduplication — Slippi rollback replays frames with duplicate/backward numbers.
  // We track the highest frame number seen and reject anything <= it.
  const lastFrameNumRef = useRef(-999);

  // FPS tracking (debug only — logged once per second)
  const fpsRef = useRef({ received: 0, played: 0, dropped: 0, lastLog: 0 });

  // Convert arena PlayerFrameData to viewer PlayerFrame
  const convertPlayerFrame = useCallback(
    (data: PlayerFrameData, matchInfo: MatchStartMessage): PlayerFrame => {
      const playerInfo = matchInfo.players.find((p) => p.port === data.port);
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
    [],
  );

  // Start smooth playback from buffer
  const startPlayback = useCallback(() => {
    if (playingRef.current) return;
    playingRef.current = true;

    console.log(
      `[useLiveMatch] Starting playback, buffer: ${frameBufferRef.current.size}`,
    );

    let lastFrameTime = performance.now();
    const targetFrameTime = PLAYBACK_INTERVAL_MS;

    const tick = (now: number) => {
      if (!playingRef.current) return;

      const elapsed = now - lastFrameTime;
      const rawFramesToPlay = Math.floor(elapsed / targetFrameTime);
      const framesToPlay = Math.min(rawFramesToPlay, 2); // Cap to avoid teleporting

      if (framesToPlay > 0) {
        lastFrameTime += framesToPlay * targetFrameTime;
        const buf = frameBufferRef.current;

        // Gradual catchup when buffer is overfull — consume extra frames per tick
        // instead of mass-dropping which causes teleporting
        let totalToConsume = framesToPlay;
        if (buf.size > OVERFLOW_THRESHOLD) {
          totalToConsume += CATCHUP_RATE;
        }

        // Consume frames
        let frameToShow: MatchFrame | undefined;
        for (let i = 0; i < totalToConsume; i++) {
          const f = ringShift(buf);
          if (f) {
            frameToShow = f;
            fpsRef.current.played++;
          }
        }

        if (frameToShow) {
          setState((prev) => ({
            ...prev,
            currentFrame: frameToShow,
            bufferHealth: Math.min(1, buf.size / BUFFER_TARGET),
          }));
        }
      }

      // FPS log once per second
      if (now - fpsRef.current.lastLog >= 1000) {
        const fps = fpsRef.current;
        console.log(
          `[useLiveMatch] rx=${fps.received}fps tx=${fps.played}fps drop=${fps.dropped} buf=${frameBufferRef.current.size}`,
        );
        fps.received = 0;
        fps.played = 0;
        fps.dropped = 0;
        fps.lastLog = now;
      }

      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  }, []);

  // Stop playback
  const stopPlayback = useCallback(() => {
    playingRef.current = false;
    frameBufferRef.current = createRing(BUFFER_CAPACITY);
    bufferingRef.current = true;
    matchInfoRef.current = null;
    lastFrameNumRef.current = -999;
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const msg: ArenaMessage = JSON.parse(event.data);

        switch (msg.type) {
          case "match_start":
            console.log(`[useLiveMatch] match_start:`, msg);
            if (wsRef.current) {
              const ws = wsRef.current as WebSocket & {
                _connectTimeout?: ReturnType<typeof setTimeout>;
              };
              if (ws._connectTimeout) {
                clearTimeout(ws._connectTimeout);
                ws._connectTimeout = undefined;
              }
            }
            matchInfoRef.current = msg;
            setState((prev) => ({
              ...prev,
              status: "connected",
              matchInfo: msg,
              error: null,
            }));
            optionsRef.current.onMatchStart?.(msg);
            break;

          case "frame": {
            const matchInfo = matchInfoRef.current;
            if (!matchInfo) break;

            fpsRef.current.received++;

            // --- Frame deduplication ---
            // Slippi rollback replays frames with duplicate/backward frame numbers.
            // Reject any frame that isn't strictly newer than what we've already seen.
            if (msg.frame <= lastFrameNumRef.current) {
              fpsRef.current.dropped++;
              break;
            }
            lastFrameNumRef.current = msg.frame;

            const players = msg.players.map((p) =>
              convertPlayerFrame(p, matchInfo),
            );

            ringPush(frameBufferRef.current, {
              frame: msg.frame,
              stageId: matchInfo.stageId,
              players,
            });

            // Start playback once buffer is full enough
            if (
              bufferingRef.current &&
              frameBufferRef.current.size >= BUFFER_TARGET
            ) {
              bufferingRef.current = false;
              startPlayback();
            }
            break;
          }

          case "game_end":
            setState((prev) => {
              const newScore: [number, number] = [...prev.gameScore];
              if (msg.winnerPort === 1) newScore[0]++;
              else newScore[1]++;
              return { ...prev, gameScore: newScore };
            });
            break;

          case "match_end":
            endedRef.current = true;
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
    [convertPlayerFrame, startPlayback],
  );

  // Connect/disconnect based on matchId — with auto-reconnect
  useEffect(() => {
    if (!matchId) {
      stopPlayback();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      reconnectAttemptRef.current = 0;
      endedRef.current = false;
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

    const MAX_RECONNECT_ATTEMPTS = 10;
    const RECONNECT_BASE_MS = 1000; // 1s, 2s, 4s... up to 30s

    function connect() {
      const wsUrl = ARENA_URL.replace(/^http/, "ws") + `/ws/match/${matchId}`;
      setState((prev) => ({ ...prev, status: "connecting", error: null }));

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`[useLiveMatch] Connected to ${wsUrl}`);
        reconnectAttemptRef.current = 0; // Reset on successful connect
        const timeout = setTimeout(() => {
          setState((prev) => {
            if (prev.status === "connecting") {
              return { ...prev, status: "ended", error: "Match has already ended" };
            }
            return prev;
          });
        }, 5000);
        (
          ws as WebSocket & { _connectTimeout?: ReturnType<typeof setTimeout> }
        )._connectTimeout = timeout;
      };

      ws.onmessage = handleMessage;

      ws.onerror = () => {
        setState((prev) => ({ ...prev, status: "error", error: "Connection error" }));
      };

      ws.onclose = (event) => {
        console.log(`[useLiveMatch] Disconnected: ${event.code} ${event.reason}`);
        if (wsRef.current !== ws) return; // Stale socket

        // Don't reconnect if match ended normally or was explicitly closed
        if (endedRef.current || event.code === 4004) {
          setState((prev) => ({ ...prev, status: "ended" }));
          return;
        }

        // Attempt reconnect with exponential backoff
        if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(
            RECONNECT_BASE_MS * Math.pow(2, reconnectAttemptRef.current),
            30_000,
          );
          reconnectAttemptRef.current++;
          console.log(
            `[useLiveMatch] Reconnecting in ${delay}ms (attempt ${reconnectAttemptRef.current}/${MAX_RECONNECT_ATTEMPTS})`,
          );
          setState((prev) => ({
            ...prev,
            status: "connecting",
            error: `Reconnecting (${reconnectAttemptRef.current}/${MAX_RECONNECT_ATTEMPTS})...`,
          }));
          reconnectTimerRef.current = setTimeout(connect, delay);
        } else {
          setState((prev) => ({
            ...prev,
            status: "disconnected",
            error: "Connection lost",
          }));
        }
      };
    }

    connect();

    return () => {
      endedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      const ws = wsRef.current;
      if (ws) {
        const wsWithTimeout = ws as WebSocket & {
          _connectTimeout?: ReturnType<typeof setTimeout>;
        };
        if (wsWithTimeout._connectTimeout) {
          clearTimeout(wsWithTimeout._connectTimeout);
        }
        stopPlayback();
        ws.close();
        wsRef.current = null;
      }
    };
  }, [matchId, handleMessage, stopPlayback]);

  return state;
}
