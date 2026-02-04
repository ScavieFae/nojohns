import { useMemo } from "react";
import { useMatchEvents } from "./useMatchEvents";
import { useWagerEvents } from "./useWagerEvents";
import type { ProtocolStats } from "../types";

export function useStats(): ProtocolStats & { isLoading: boolean } {
  const { data: matches, isLoading: matchesLoading } = useMatchEvents();
  const { data: wagers, isLoading: wagersLoading } = useWagerEvents();

  const stats = useMemo(() => {
    const totalMatches = matches?.length ?? 0;

    const totalWagered = wagers?.reduce((sum, w) => sum + w.amount, 0n) ?? 0n;

    const agents = new Set<string>();
    for (const m of matches ?? []) {
      agents.add(m.winner);
      agents.add(m.loser);
    }

    return {
      totalMatches,
      totalWagered,
      uniqueAgents: agents.size,
    };
  }, [matches, wagers]);

  return { ...stats, isLoading: matchesLoading || wagersLoading };
}
