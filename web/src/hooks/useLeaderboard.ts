import { useMemo } from "react";
import { useMatchEvents } from "./useMatchEvents";
import type { AgentStats, MatchRecord } from "../types";

const STARTING_ELO = 1500;
const K_FACTOR = 32;

/**
 * Calculate expected score for player A against player B
 * Formula: 1 / (1 + 10^((Rb - Ra) / 400))
 */
function expectedScore(eloA: number, eloB: number): number {
  return 1 / (1 + Math.pow(10, (eloB - eloA) / 400));
}

/**
 * Calculate new Elo rating after a match
 * @param elo Current Elo rating
 * @param expected Expected score (0-1)
 * @param actual Actual score (1 for win, 0 for loss)
 */
function newElo(elo: number, expected: number, actual: number): number {
  return Math.round(elo + K_FACTOR * (actual - expected));
}

/**
 * Compute Elo ratings from match history
 * Processes matches chronologically to compute accurate ratings
 */
function computeEloRatings(
  matches: MatchRecord[]
): Map<`0x${string}`, number> {
  const ratings = new Map<`0x${string}`, number>();

  // Sort by timestamp (oldest first)
  const sorted = [...matches].sort((a, b) => {
    const diff = a.timestamp - b.timestamp;
    return diff < 0n ? -1 : diff > 0n ? 1 : 0;
  });

  for (const match of sorted) {
    const winnerElo = ratings.get(match.winner) ?? STARTING_ELO;
    const loserElo = ratings.get(match.loser) ?? STARTING_ELO;

    const expectedWin = expectedScore(winnerElo, loserElo);
    const expectedLoss = expectedScore(loserElo, winnerElo);

    ratings.set(match.winner, newElo(winnerElo, expectedWin, 1));
    ratings.set(match.loser, newElo(loserElo, expectedLoss, 0));
  }

  return ratings;
}

export function useLeaderboard() {
  const { data: matches, ...rest } = useMatchEvents();

  const leaderboard = useMemo((): AgentStats[] => {
    if (!matches) return [];

    // Compute Elo ratings from match history
    const eloRatings = computeEloRatings(matches);

    // Aggregate W/L stats
    const stats = new Map<`0x${string}`, { wins: number; losses: number }>();

    for (const match of matches) {
      const w = stats.get(match.winner) ?? { wins: 0, losses: 0 };
      w.wins++;
      stats.set(match.winner, w);

      const l = stats.get(match.loser) ?? { wins: 0, losses: 0 };
      l.losses++;
      stats.set(match.loser, l);
    }

    return Array.from(stats.entries())
      .map(([address, { wins, losses }]) => ({
        address,
        wins,
        losses,
        totalMatches: wins + losses,
        winRate: wins + losses > 0 ? wins / (wins + losses) : 0,
        elo: eloRatings.get(address) ?? STARTING_ELO,
      }))
      .sort((a, b) => b.elo - a.elo); // Sort by Elo (highest first)
  }, [matches]);

  return { data: leaderboard, ...rest };
}
