/**
 * Hook for connecting to live match WebSocket stream
 */

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  ArenaMessage,
  MatchStartMessage,
  PlayerFrameData,
} from "../lib/liveMatchProtocol";
import type { MatchFrame, PlayerFrame } from "../components/viewer/MeleeViewer";
import { externalIdByInternalId } from "../lib/meleeIds";
import { ARENA_URL } from "../config";

export interface LiveMatchState {
  status: "connecting" | "connected" | "ended" | "error" | "disconnected";
  matchInfo: MatchStartMessage | null;
  currentFrame: MatchFrame | null;
  error: string | null;
  gameScore: [number, number];
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
  });

  const wsRef = useRef<WebSocket | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Convert arena PlayerFrameData to viewer PlayerFrame
  const convertPlayerFrame = useCallback(
    (data: PlayerFrameData, matchInfo: MatchStartMessage): PlayerFrame => {
      // Find player info from match start
      const playerInfo = matchInfo.players.find((p) => p.port === data.port);
      const externalCharId = playerInfo?.characterId ?? 0;

      // Map external char ID to internal for animation lookup
      // (viewer uses internal IDs for animation cache keying)
      const internalCharId =
        Object.entries(externalIdByInternalId).find(
          ([, ext]) => ext === externalCharId
        )?.[0] ?? "1";

      return {
        internalCharId: parseInt(internalCharId, 10),
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

  // Handle incoming messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const msg: ArenaMessage = JSON.parse(event.data);

        // Debug logging for all messages
        if (msg.type === "frame") {
          // Only log every 60th frame to avoid spam
          if ((msg as { frame: number }).frame % 60 === 0) {
            console.log(`[useLiveMatch] Frame ${(msg as { frame: number }).frame}`);
          }
        } else {
          console.log(`[useLiveMatch] Received message:`, msg.type, msg);
        }

        switch (msg.type) {
          case "match_start":
            // Clear the connect timeout since we got match info
            if (wsRef.current) {
              const ws = wsRef.current as WebSocket & { _connectTimeout?: ReturnType<typeof setTimeout> };
              if (ws._connectTimeout) {
                clearTimeout(ws._connectTimeout);
                ws._connectTimeout = undefined;
              }
            }
            setState((prev) => ({
              ...prev,
              status: "connected",
              matchInfo: msg,
              error: null,
            }));
            optionsRef.current.onMatchStart?.(msg);
            break;

          case "frame":
            setState((prev) => {
              if (!prev.matchInfo) return prev;

              const players = msg.players.map((p) =>
                convertPlayerFrame(p, prev.matchInfo!)
              );

              return {
                ...prev,
                currentFrame: {
                  frame: msg.frame,
                  stageId: prev.matchInfo.stageId,
                  players,
                },
              };
            });
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
    [convertPlayerFrame]
  );

  // Connect/disconnect based on matchId
  useEffect(() => {
    if (!matchId) {
      // Disconnect
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
      ws.close();
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, [matchId, handleMessage]);

  return state;
}
