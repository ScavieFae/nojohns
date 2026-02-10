/**
 * Animation cache for Melee character animations
 * Loads animation ZIPs from /zips/ and caches them in memory
 *
 * Animation data source: SlippiLab (MIT) - https://github.com/frankborden/slippilab
 */

import { unzipSync } from "fflate";
import { characterZipByExternalId } from "./meleeIds";

// Animation name -> array of SVG path strings (one per frame)
export type CharacterAnimations = Record<string, string[]>;

const cache = new Map<number, CharacterAnimations>();
const loading = new Map<number, Promise<CharacterAnimations | null>>();

/**
 * Fetch and cache animations for a character by external ID.
 * Returns null on failure (caller should retry).
 */
export async function fetchAnimations(externalCharId: number): Promise<CharacterAnimations | null> {
  // Already cached
  const cached = cache.get(externalCharId);
  if (cached) return cached;

  // Already loading
  const inFlight = loading.get(externalCharId);
  if (inFlight) return inFlight;

  // Start loading
  const zipName = characterZipByExternalId[externalCharId];
  if (!zipName) {
    console.warn(`[anim] No ZIP for character ${externalCharId}`);
    return null;
  }

  console.log(`[anim] Loading ${zipName}.zip (externalId=${externalCharId})`);

  const promise = (async () => {
    try {
      const response = await fetch(`/zips/${zipName}.zip`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const unzipped = unzipSync(new Uint8Array(arrayBuffer));

      const animations: CharacterAnimations = {};
      for (const [filename, data] of Object.entries(unzipped)) {
        if (!filename.endsWith(".json")) continue;
        const animName = filename.replace(".json", "");
        const text = new TextDecoder().decode(data);
        animations[animName] = JSON.parse(text);
      }

      cache.set(externalCharId, animations);
      console.log(`[anim] Loaded ${zipName}: ${Object.keys(animations).length} animations`);
      return animations;
    } catch (err) {
      console.error(`[anim] Failed to load ${zipName}.zip:`, err);
      return null;
    } finally {
      loading.delete(externalCharId);
    }
  })();

  loading.set(externalCharId, promise);
  return promise;
}

/**
 * Get cached animations (returns undefined if not loaded)
 */
export function getAnimations(externalCharId: number): CharacterAnimations | undefined {
  return cache.get(externalCharId);
}

/**
 * Preload animations for multiple characters
 */
export async function preloadAnimations(externalCharIds: number[]): Promise<void> {
  await Promise.all(externalCharIds.map(fetchAnimations));
}
