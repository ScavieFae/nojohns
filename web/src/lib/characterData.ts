/**
 * Character-specific data for rendering
 * Source: SlippiLab (MIT) - https://github.com/frankborden/slippilab
 */

export interface CharacterData {
  scale: number;
  shieldOffset: [number, number];
  shieldSize: number;
}

// Character data indexed by external character ID
// Most characters use similar values; these are approximations
export const characterDataByExternalId: Record<number, CharacterData> = {
  0:  { scale: 1.12, shieldOffset: [0, 8], shieldSize: 13.5 },  // Captain Falcon
  1:  { scale: 1.30, shieldOffset: [0, 10], shieldSize: 16.0 }, // Donkey Kong
  2:  { scale: 0.96, shieldOffset: [2.7, 9], shieldSize: 13.8 }, // Fox
  3:  { scale: 0.85, shieldOffset: [0, 7], shieldSize: 11.0 },  // Mr. Game & Watch
  4:  { scale: 0.82, shieldOffset: [0, 6], shieldSize: 10.0 },  // Kirby
  5:  { scale: 0.90, shieldOffset: [0, 12], shieldSize: 18.0 }, // Bowser (big but not GIANT)
  6:  { scale: 1.05, shieldOffset: [0, 9], shieldSize: 13.0 },  // Link
  7:  { scale: 0.95, shieldOffset: [0, 8], shieldSize: 12.5 },  // Luigi
  8:  { scale: 0.95, shieldOffset: [0, 8], shieldSize: 12.5 },  // Mario
  9:  { scale: 1.05, shieldOffset: [0, 9], shieldSize: 14.0 },  // Marth
  10: { scale: 1.30, shieldOffset: [0, 10], shieldSize: 15.0 }, // Mewtwo
  11: { scale: 0.90, shieldOffset: [0, 7], shieldSize: 11.0 },  // Ness
  12: { scale: 0.95, shieldOffset: [0, 8], shieldSize: 12.5 },  // Peach
  13: { scale: 0.85, shieldOffset: [0, 6], shieldSize: 10.5 },  // Pikachu
  14: { scale: 0.90, shieldOffset: [0, 7], shieldSize: 11.0 },  // Ice Climbers
  15: { scale: 0.70, shieldOffset: [0, 5], shieldSize: 9.0 },   // Jigglypuff
  16: { scale: 1.10, shieldOffset: [0, 9], shieldSize: 14.0 },  // Samus
  17: { scale: 1.00, shieldOffset: [0, 8], shieldSize: 12.0 },  // Yoshi
  18: { scale: 1.00, shieldOffset: [0, 8], shieldSize: 12.5 },  // Zelda
  19: { scale: 0.96, shieldOffset: [0, 8], shieldSize: 12.0 },  // Sheik
  20: { scale: 0.98, shieldOffset: [2.5, 9], shieldSize: 13.5 }, // Falco
  21: { scale: 1.05, shieldOffset: [0, 8], shieldSize: 11.5 },  // Young Link (slightly bigger)
  22: { scale: 0.95, shieldOffset: [0, 8], shieldSize: 12.5 },  // Dr. Mario
  23: { scale: 1.05, shieldOffset: [0, 9], shieldSize: 14.0 },  // Roy
  24: { scale: 0.55, shieldOffset: [0, 4], shieldSize: 7.5 },   // Pichu (tiny!)
  25: { scale: 1.20, shieldOffset: [0, 9], shieldSize: 14.5 },  // Ganondorf
};

// Common animation name remappings that apply to most characters
export const commonAnimationRemaps: Record<string, string> = {
  "AppealL": "Appeal",
  "AppealR": "Appeal",
  "Escape": "EscapeN",
  "GuardReflect": "Guard",
  "GuardSetOff": "GuardDamage",
  "KneeBend": "Landing",
  "LandingFallSpecial": "Landing",
  "CatchDashPull": "CatchWait",
  "CatchPull": "CatchWait",
  "EntryEnd": "Entry",
  "EntryStart": "Entry",
  "Wait": "Wait1",
};

/**
 * Get the animation name to look up for a given action
 */
export function getAnimationName(actionName: string): string {
  return commonAnimationRemaps[actionName] ?? actionName;
}
