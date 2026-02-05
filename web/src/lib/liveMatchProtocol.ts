/**
 * Live Match WebSocket Protocol
 *
 * This defines the data format for streaming live match frames from arena to viewers.
 * Arena (Python/FastAPI) sends JSON messages over WebSocket at 60fps during active matches.
 *
 * Endpoint: ws://{arena_host}/ws/match/{match_id}
 *
 * For Scav: implement this in arena/server.py
 */

// ============================================================================
// MESSAGES FROM ARENA → VIEWER
// ============================================================================

/**
 * Match started - sent once when connection established during active match
 */
export interface MatchStartMessage {
  type: "match_start";
  matchId: string;
  stageId: number;           // External stage ID (2=FoD, 3=Pokemon, 8=Yoshi's, etc.)
  players: {
    port: number;            // 1-4
    characterId: number;     // External character ID (0=Falcon, 2=Fox, 9=Marth, etc.)
    connectCode: string;     // e.g. "ABCD#123"
    displayName?: string;
  }[];
}

/**
 * Frame update - sent every frame (~60fps) during gameplay
 * This is the hot path - keep it minimal
 */
export interface FrameMessage {
  type: "frame";
  frame: number;             // Frame number (starts at -123 for pre-game, 0+ for gameplay)
  players: PlayerFrameData[];
}

export interface PlayerFrameData {
  port: number;              // 1-4 (matches player index from match_start)

  // Position (game units, origin at stage center)
  x: number;
  y: number;

  // Animation state
  actionStateId: number;     // Action state ID (14=Wait, 20=Dash, 65=Nair, etc.)
  actionFrame: number;       // Frame within current action (float, can be fractional)
  facingDirection: number;   // 1 = right, -1 = left

  // Game state
  percent: number;           // Damage percent (0-999)
  stocks: number;            // Remaining stocks

  // Optional: for shield rendering
  shieldHealth?: number;     // 0-60, only sent when shielding

  // Optional: for hitstun/invincibility coloring
  isInvincible?: boolean;
  isInHitstun?: boolean;
}

/**
 * Game end - sent when a game in the set ends
 */
export interface GameEndMessage {
  type: "game_end";
  gameNumber: number;        // 1-indexed game in the set
  winnerPort: number;        // Port of winner (1-4)
  endMethod: "stocks" | "timeout" | "lras";  // How game ended
}

/**
 * Match end - sent when the full match/set is complete
 */
export interface MatchEndMessage {
  type: "match_end";
  winnerPort: number;
  finalScore: [number, number];  // Games won by each player
}

/**
 * Error - sent on connection issues or invalid state
 */
export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ArenaMessage =
  | MatchStartMessage
  | FrameMessage
  | GameEndMessage
  | MatchEndMessage
  | ErrorMessage;


// ============================================================================
// IMPLEMENTATION NOTES FOR SCAV (arena/server.py)
// ============================================================================
/*

1. WebSocket endpoint: /ws/match/{match_id}
   - Accept connections from spectators
   - If match is in progress, send match_start immediately, then frames
   - If match not found or ended, send error and close

2. Frame streaming from libmelee:
   - In the game loop, after `console.step()`, extract player state
   - Broadcast FrameMessage to all connected WebSocket clients
   - Use asyncio for non-blocking broadcast

3. Data extraction from libmelee GameState:

   ```python
   def extract_player_frame(player: PlayerState, port: int) -> dict:
       return {
           "port": port,
           "x": player.position.x,
           "y": player.position.y,
           "actionStateId": player.action.value,
           "actionFrame": player.action_frame,
           "facingDirection": 1 if player.facing else -1,
           "percent": player.percent,
           "stocks": player.stock,
           "shieldHealth": player.shield_strength if player.action.value in [178, 179, 180, 182] else None,
           "isInvincible": player.invulnerable,
           "isInHitstun": player.hitstun_frames_left > 0,
       }
   ```

4. Bandwidth considerations:
   - FrameMessage is ~100-200 bytes JSON
   - At 60fps = ~6-12 KB/s per viewer
   - Consider throttling to 30fps for many viewers (every other frame)
   - Consider binary format (MessagePack) if scaling to 100+ viewers

5. Character ID mapping:
   - libmelee uses internal IDs (0=Mario, 1=Fox, 2=Falcon...)
   - Convert to external IDs for the viewer using:
     external_by_internal = {0:8, 1:2, 2:0, 3:1, 4:4, 5:5, ...}
   - Or send internal ID and let viewer convert (viewer has the mapping)

*/

// ============================================================================
// HELPER: Convert internal character ID to external
// ============================================================================
export const externalCharIdByInternal: Record<number, number> = {
  0: 8,   // Mario
  1: 2,   // Fox
  2: 0,   // Captain Falcon
  3: 1,   // Donkey Kong
  4: 4,   // Kirby
  5: 5,   // Bowser
  6: 6,   // Link
  7: 19,  // Sheik
  8: 11,  // Ness
  9: 12,  // Peach
  10: 14, // Popo (Ice Climbers)
  11: 14, // Nana
  12: 13, // Pikachu
  13: 16, // Samus
  14: 17, // Yoshi
  15: 15, // Jigglypuff
  16: 10, // Mewtwo
  17: 7,  // Luigi
  18: 9,  // Marth
  19: 18, // Zelda
  20: 21, // Young Link
  21: 22, // Dr. Mario
  22: 20, // Falco
  23: 24, // Pichu
  24: 3,  // Mr. Game & Watch
  25: 25, // Ganondorf
  26: 23, // Roy
};
