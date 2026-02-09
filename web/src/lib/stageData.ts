/**
 * Melee stage platform data
 *
 * Coordinates are in game units. Y=0 is the main stage surface.
 * Platform positions from libmelee/Slippi documentation.
 */

export interface Platform {
  x: number;      // center x
  y: number;      // top surface y
  width: number;  // total width
  height: number; // visual thickness
}

export interface StageData {
  name: string;
  mainPlatform: Platform;
  platforms: Platform[]; // side/top platforms
  blastzones: {
    left: number;
    right: number;
    top: number;
    bottom: number;
  };
  backgroundColor?: string;
}

// Stage IDs from libmelee (internal stage IDs)
export const STAGE_IDS = {
  FOUNTAIN_OF_DREAMS: 2,
  POKEMON_STADIUM: 3,
  YOSHIS_STORY: 8,
  DREAMLAND: 28,
  BATTLEFIELD: 31,
  FINAL_DESTINATION: 32,
} as const;

// Legal stage data (tournament legal stages)
export const stageDataById: Record<number, StageData> = {
  // Battlefield
  [STAGE_IDS.BATTLEFIELD]: {
    name: "Battlefield",
    mainPlatform: { x: 0, y: 0, width: 79.0, height: 6 },
    platforms: [
      { x: -57.6, y: 27.2, width: 51.0, height: 3 }, // left
      { x: 57.6, y: 27.2, width: 51.0, height: 3 },  // right
      { x: 0, y: 54.4, width: 51.0, height: 3 },     // top
    ],
    blastzones: { left: -224, right: 224, top: 200, bottom: -108.8 },
    backgroundColor: "#1a1a2e",
  },

  // Final Destination
  [STAGE_IDS.FINAL_DESTINATION]: {
    name: "Final Destination",
    mainPlatform: { x: 0, y: 0, width: 85.5, height: 6 },
    platforms: [], // No platforms!
    blastzones: { left: -246, right: 246, top: 188, bottom: -140 },
    backgroundColor: "#0f0f1a",
  },

  // Dreamland N64
  [STAGE_IDS.DREAMLAND]: {
    name: "Dreamland",
    mainPlatform: { x: 0, y: 0, width: 77.3, height: 6 },
    platforms: [
      { x: -61.4, y: 30.2, width: 31.5, height: 3 }, // left
      { x: 61.4, y: 30.2, width: 31.5, height: 3 },  // right
      { x: 0, y: 51.4, width: 31.5, height: 3 },     // top
    ],
    blastzones: { left: -255, right: 255, top: 250, bottom: -123 },
    backgroundColor: "#1e3a5f",
  },

  // Yoshi's Story
  [STAGE_IDS.YOSHIS_STORY]: {
    name: "Yoshi's Story",
    mainPlatform: { x: 0, y: 0, width: 56.0, height: 6 },
    platforms: [
      { x: -59.5, y: 23.5, width: 24.0, height: 3 }, // left
      { x: 59.5, y: 23.5, width: 24.0, height: 3 },  // right
      { x: 0, y: 42.0, width: 28.0, height: 3 },     // top (Randall's path)
    ],
    blastzones: { left: -175.7, right: 173.6, top: 168, bottom: -91 },
    backgroundColor: "#2d4a3e",
  },

  // Fountain of Dreams
  [STAGE_IDS.FOUNTAIN_OF_DREAMS]: {
    name: "Fountain of Dreams",
    mainPlatform: { x: 0, y: 0, width: 63.0, height: 6 },
    platforms: [
      { x: -49.5, y: 16.5, width: 21.0, height: 3 }, // left (moves)
      { x: 49.5, y: 16.5, width: 21.0, height: 3 },  // right (moves)
      { x: 0, y: 42.75, width: 28.0, height: 3 },    // top
    ],
    blastzones: { left: -198.75, right: 198.75, top: 202.5, bottom: -146.25 },
    backgroundColor: "#1a2a4a",
  },

  // Pokemon Stadium
  [STAGE_IDS.POKEMON_STADIUM]: {
    name: "Pokemon Stadium",
    mainPlatform: { x: 0, y: 0, width: 87.75, height: 6 },
    platforms: [
      { x: -55, y: 25, width: 25.0, height: 3 }, // left
      { x: 55, y: 25, width: 25.0, height: 3 },  // right
    ],
    blastzones: { left: -230, right: 230, top: 180, bottom: -111 },
    backgroundColor: "#2a2a3a",
  },
};

// Default stage if ID not found
export const defaultStage: StageData = {
  name: "Unknown Stage",
  mainPlatform: { x: 0, y: 0, width: 80, height: 6 },
  platforms: [],
  blastzones: { left: -200, right: 200, top: 180, bottom: -100 },
  backgroundColor: "#1a1a1a",
};

export function getStageData(stageId: number): StageData {
  return stageDataById[stageId] ?? defaultStage;
}
