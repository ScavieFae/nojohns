/**
 * Parse .slp replay files for playback in the viewer
 * Uses @slippi/slippi-js
 */

import { SlippiGame } from "@slippi/slippi-js";
import type { MatchFrame, PlayerFrame } from "../components/viewer/MeleeViewer";

export interface ParsedReplay {
  settings: {
    stageId: number;
    players: {
      port: number;
      characterId: number;  // External character ID
      displayName: string;
    }[];
  };
  frames: MatchFrame[];
  totalFrames: number;
}

/**
 * Parse a .slp file buffer into viewer-compatible frames
 */
export function parseReplay(buffer: ArrayBuffer): ParsedReplay {
  const game = new SlippiGame(buffer);

  const settings = game.getSettings();
  if (!settings) {
    throw new Error("Failed to read game settings from replay");
  }

  const frames = game.getFrames();
  if (!frames) {
    throw new Error("Failed to read frames from replay");
  }

  // Extract player info - keep track of playerIndex for frame lookup
  const players = settings.players
    .filter((p) => p.type !== 3) // Filter out empty slots (type 3)
    .map((p) => ({
      port: p.port + 1, // slippi-js uses 0-indexed ports, display as 1-indexed
      playerIndex: p.playerIndex, // The index in frame.players array
      characterId: p.characterId ?? 0,
      displayName: p.displayName ?? p.connectCode ?? `P${p.port + 1}`,
    }));

  // Convert frames to viewer format
  // Slippi frames are keyed by frame number, starting at -123 (pre-game)
  // We only care about frames >= 0 (actual gameplay)
  const frameNumbers = Object.keys(frames)
    .map(Number)
    .filter((n) => n >= 0)
    .sort((a, b) => a - b);

  const parsedFrames: MatchFrame[] = frameNumbers.map((frameNum) => {
    const frame = frames[frameNum];

    const playerFrames: PlayerFrame[] = players.map((playerInfo) => {
      // Use playerIndex to look up frame data, not port number
      const playerFrame = frame.players[playerInfo.playerIndex];

      if (!playerFrame?.post) {
        // Player might be dead or not present
        return {
          internalCharId: externalToInternal(playerInfo.characterId),
          x: 0,
          y: -1000, // Off-screen
          actionStateId: 0, // DeadDown
          actionFrame: 0,
          facingDirection: 1,
          percent: 0,
          stocks: 0,
        };
      }

      const post = playerFrame.post;

      return {
        internalCharId: post.internalCharacterId ?? externalToInternal(playerInfo.characterId),
        x: post.positionX ?? 0,
        y: post.positionY ?? 0,
        actionStateId: post.actionStateId ?? 14,
        actionFrame: post.actionStateCounter ?? 0,
        facingDirection: post.facingDirection ?? 1,
        percent: post.percent ?? 0,
        stocks: post.stocksRemaining ?? 0,
      };
    });

    return {
      frame: frameNum,
      stageId: settings.stageId ?? 8,
      players: playerFrames,
    };
  });

  return {
    settings: {
      stageId: settings.stageId ?? 8,
      players,
    },
    frames: parsedFrames,
    totalFrames: parsedFrames.length,
  };
}

/**
 * Convert external character ID to internal character ID
 * (External is used in settings, internal is used in frame data and animations)
 */
function externalToInternal(externalId: number): number {
  const mapping: Record<number, number> = {
    0: 2,   // Captain Falcon
    1: 3,   // Donkey Kong
    2: 1,   // Fox
    3: 24,  // Mr. Game & Watch
    4: 4,   // Kirby
    5: 5,   // Bowser
    6: 6,   // Link
    7: 17,  // Luigi
    8: 0,   // Mario
    9: 18,  // Marth
    10: 16, // Mewtwo
    11: 8,  // Ness
    12: 9,  // Peach
    13: 12, // Pikachu
    14: 10, // Ice Climbers (Popo)
    15: 15, // Jigglypuff
    16: 13, // Samus
    17: 14, // Yoshi
    18: 19, // Zelda
    19: 7,  // Sheik
    20: 22, // Falco
    21: 20, // Young Link
    22: 21, // Dr. Mario
    23: 26, // Roy
    24: 23, // Pichu
    25: 25, // Ganondorf
  };
  return mapping[externalId] ?? 0;
}

/**
 * Load a replay file from a File object (e.g., from <input type="file">)
 */
export async function loadReplayFile(file: File): Promise<ParsedReplay> {
  const buffer = await file.arrayBuffer();
  return parseReplay(buffer);
}
