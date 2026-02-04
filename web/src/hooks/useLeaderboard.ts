import { useMemo } from "react";
import { useMatchEvents } from "./useMatchEvents";
import type { AgentStats } from "../types";

export function useLeaderboard() {
  const { data: matches, ...rest } = useMatchEvents();

  const leaderboard = useMemo((): AgentStats[] => {
    if (!matches) return [];

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
      }))
      .sort((a, b) => b.wins - a.wins || a.losses - b.losses);
  }, [matches]);

  return { data: leaderboard, ...rest };
}
