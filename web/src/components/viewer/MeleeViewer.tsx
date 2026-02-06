/**
 * Melee match viewer using SlippiLab animation data
 * Renders characters as SVG paths based on frame data
 *
 * Animation data: SlippiLab (MIT) - https://github.com/frankborden/slippilab
 */

import { useEffect, useState, useMemo } from "react";
import { fetchAnimations, getAnimations, type CharacterAnimations } from "../../lib/animationCache";
import { actionNameById, externalIdByInternalId } from "../../lib/meleeIds";
import { characterDataByExternalId, getAnimationName } from "../../lib/characterData";

// Frame data from arena WebSocket
export interface PlayerFrame {
  internalCharId: number;
  x: number;
  y: number;
  actionStateId: number;
  actionFrame: number;
  facingDirection: number; // 1 = right, -1 = left
  percent: number;
  stocks: number;
}

export interface MatchFrame {
  frame: number;
  stageId: number;
  players: PlayerFrame[];
}

interface MeleeViewerProps {
  frame: MatchFrame | null;
}

// Player colors (port 1-4) - brand colors
const PLAYER_COLORS = ["#22c55e", "#a855f7", "#3b82f6", "#f59e0b"]; // green, purple, blue, amber

function PlayerRenderer({
  player,
  playerIndex,
  animations,
}: {
  player: PlayerFrame;
  playerIndex: number;
  animations: CharacterAnimations | undefined;
}) {
  const externalCharId = externalIdByInternalId[player.internalCharId] ?? 0;
  const charData = characterDataByExternalId[externalCharId] ?? characterDataByExternalId[2];

  // Get action name and animation
  const actionName = actionNameById[player.actionStateId] ?? "Wait";
  const animName = getAnimationName(actionName);
  const animFrames = animations?.[animName] ?? animations?.["Wait1"] ?? [];

  // Get the right frame (with wraparound)
  const frameIndex = Math.floor(Math.max(0, player.actionFrame)) % Math.max(1, animFrames.length);
  let path = animFrames[frameIndex];

  // Handle frame references like "frame20"
  if (path?.startsWith("frame")) {
    const refIndex = parseInt(path.slice(5), 10);
    path = animFrames[refIndex];
  }

  if (!path) {
    // Fallback: draw a simple circle if no animation found
    return (
      <circle
        cx={player.x}
        cy={player.y}
        r={5}
        fill={PLAYER_COLORS[playerIndex]}
        opacity={0.8}
      />
    );
  }

  // Build transform: translate to position, scale by character scale, flip by facing direction
  // The SVG paths are centered around (500, 500) and need to be scaled down
  const transforms = [
    `translate(${player.x} ${player.y})`,
    `scale(${charData.scale} ${charData.scale})`,
    `scale(${player.facingDirection} 1)`,
    "scale(0.1 -0.1)",
    "translate(-500 -500)",
  ].join(" ");

  return (
    <path
      d={path}
      transform={transforms}
      fill={PLAYER_COLORS[playerIndex]}
      stroke="black"
      strokeWidth={0.5}
      opacity={0.9}
    />
  );
}

export function MeleeViewer({ frame }: MeleeViewerProps) {
  const [loadedChars, setLoadedChars] = useState<Set<number>>(new Set());

  // Load animations for characters in the frame
  useEffect(() => {
    if (!frame) return;

    const charIds = new Set<number>();
    for (const player of frame.players) {
      const externalId = externalIdByInternalId[player.internalCharId];
      if (externalId !== undefined && !loadedChars.has(externalId)) {
        charIds.add(externalId);
      }
    }

    if (charIds.size === 0) return;

    // Load missing character animations
    Promise.all([...charIds].map(fetchAnimations)).then(() => {
      setLoadedChars((prev) => {
        const next = new Set(prev);
        charIds.forEach((id) => next.add(id));
        return next;
      });
    });
  }, [frame, loadedChars]);

  // Get animations for each player
  const playerAnimations = useMemo(() => {
    if (!frame) return [];
    return frame.players.map((player) => {
      const externalId = externalIdByInternalId[player.internalCharId];
      return externalId !== undefined ? getAnimations(externalId) : undefined;
    });
  }, [frame, loadedChars]);

  if (!frame) {
    return (
      <div
        className="bg-surface-800 rounded-lg flex items-center justify-center w-full aspect-[73/60]"
      >
        <span className="text-gray-500">Waiting for match data...</span>
      </div>
    );
  }

  return (
    <div className="bg-surface-800 rounded-lg overflow-hidden w-full">
      <svg
        viewBox="-140 -120 280 200"
        className="w-full h-auto bg-surface-900"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Coordinate system: Y-axis inverted (positive = up) */}
        {/* ViewBox: x=-140..140, y=-120..80; after Y-flip shows game y=-80..120 */}
        <g transform="scale(1 -1)">
          {/* Stage platform (simplified - Yoshi's Story main platform) */}
          <rect
            x={-56}
            y={-5}
            width={112}
            height={6}
            fill="#3a3a3a"
            stroke="#555"
            strokeWidth={0.5}
          />

          {/* Players */}
          {frame.players.map((player, i) => (
            <PlayerRenderer
              key={i}
              player={player}
              playerIndex={i}
              animations={playerAnimations[i]}
            />
          ))}
        </g>

        {/* HUD overlay (not inverted) */}
        <g>
          {/* Frame counter */}
          <text x={-135} y={-110} fill="#888" fontSize={6} fontFamily="monospace">
            Frame {frame.frame}
          </text>

          {/* Player stocks/percent */}
          {frame.players.map((player, i) => (
            <g key={i} transform={`translate(${-130 + i * 100}, 72)`}>
              <text fill={PLAYER_COLORS[i]} fontSize={7} fontFamily="monospace" fontWeight="bold">
                P{i + 1}
              </text>
              <text x={15} fill="#fff" fontSize={7} fontFamily="monospace">
                {player.percent.toFixed(0)}%
              </text>
              <text x={45} fill="#888" fontSize={6} fontFamily="monospace">
                {"●".repeat(player.stocks)}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}
