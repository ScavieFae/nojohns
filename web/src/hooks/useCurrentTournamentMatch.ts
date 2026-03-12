import { useQuery } from "@tanstack/react-query";
import { ARENA_URL } from "../config";

export interface TournamentEntry {
  name: string;
  character: string;
  strategy: string;
  connect_code: string;
  wallet_address: string | null;
}

export interface CurrentMatch {
  tournament_id: string;
  tournament_name: string;
  round: number;
  slot: number;
  status: string;
  entry_a: TournamentEntry;
  entry_b: TournamentEntry;
  arena_match_id: string | null;
  pool_id: number | null;
}

async function fetchCurrentMatch(): Promise<CurrentMatch | null> {
  // 1. Find the active tournament
  const listRes = await fetch(`${ARENA_URL}/tournaments`);
  if (!listRes.ok) return null;
  const { tournaments } = await listRes.json();

  const active = (tournaments as { id: string; status: string; name: string }[]).find(
    (t) => t.status === "active"
  );
  if (!active) return null;

  // 2. Get bracket and find the playing (or next pending) match
  const bracketRes = await fetch(`${ARENA_URL}/tournaments/${active.id}/bracket`);
  if (!bracketRes.ok) return null;
  const data = await bracketRes.json();

  const rounds: Record<string, unknown>[][] = data.bracket?.rounds ?? [];
  for (const round of rounds) {
    for (const match of round) {
      if (
        match.status === "playing" &&
        match.entry_a &&
        match.entry_b
      ) {
        return {
          tournament_id: active.id,
          tournament_name: data.name,
          round: match.round as number,
          slot: match.slot as number,
          status: match.status as string,
          entry_a: match.entry_a as TournamentEntry,
          entry_b: match.entry_b as TournamentEntry,
          arena_match_id: (match.arena_match_id as string) ?? null,
          pool_id: (match.pool_id as number) ?? null,
        };
      }
    }
  }

  return null;
}

/**
 * Polls arena every 5s for the current "playing" match in the active tournament.
 * Returns null if no active match.
 */
export function useCurrentTournamentMatch() {
  const query = useQuery({
    queryKey: ["currentTournamentMatch"],
    queryFn: fetchCurrentMatch,
    refetchInterval: 5_000,
    staleTime: 3_000,
    retry: false,
  });

  return {
    match: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error,
  };
}
