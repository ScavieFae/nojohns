import { useQuery } from "@tanstack/react-query";
import { ARENA_URL } from "../config";

export interface TournamentEntry {
  name: string;
  character: string;
  strategy: string;
  connect_code: string;
  wallet_address: string | null;
  registrant: string | null;
  email: string | null;
}

export interface TournamentMatch {
  round: number;
  slot: number;
  entry_a: TournamentEntry | null;
  entry_b: TournamentEntry | null;
  winner: TournamentEntry | null;
  score_a: number | null;
  score_b: number | null;
  pool_id: number | null;
  status: string;
  arena_match_id: string | null;
}

export interface TournamentData {
  id: string;
  name: string;
  status: "registration" | "pending" | "active" | "complete";
  featured: boolean;
  entries: TournamentEntry[];
  bracket: {
    size: number;
    rounds: TournamentMatch[][];
  };
  current_round: number;
  current_slot: number;
  champion?: TournamentEntry | null;
}

async function fetchFeaturedTournament(): Promise<TournamentData | null> {
  try {
    const res = await fetch(`${ARENA_URL}/tournaments/featured`);
    if (!res.ok) return null;
    const data = await res.json();
    // Derive champion from bracket
    const rounds = data.bracket?.rounds ?? [];
    if (rounds.length > 0) {
      const final = rounds[rounds.length - 1];
      if (final.length === 1 && final[0].winner) {
        data.champion = final[0].winner;
      }
    }
    return data;
  } catch {
    return null;
  }
}

export function useFeaturedTournament() {
  const query = useQuery({
    queryKey: ["featuredTournament"],
    queryFn: fetchFeaturedTournament,
    refetchInterval: 5_000,
    staleTime: 3_000,
  });

  return {
    tournament: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error,
  };
}

async function fetchMyEntry(
  tournamentId: string,
  email: string | null,
  wallet: string | null,
): Promise<TournamentEntry | null> {
  if (!email && !wallet) return null;
  const params = new URLSearchParams();
  if (email) params.set("email", email);
  if (wallet) params.set("wallet", wallet);
  try {
    const res = await fetch(
      `${ARENA_URL}/tournaments/${tournamentId}/my-entry?${params}`,
    );
    if (!res.ok) return null;
    const data = await res.json();
    return data.found ? data.entry : null;
  } catch {
    return null;
  }
}

export function useMyEntry(
  tournamentId: string | undefined,
  email: string | null,
  wallet: string | null,
) {
  const query = useQuery({
    queryKey: ["myEntry", tournamentId, email, wallet],
    queryFn: () => fetchMyEntry(tournamentId!, email, wallet),
    enabled: !!tournamentId && (!!email || !!wallet),
    staleTime: 10_000,
  });

  return {
    entry: query.data ?? null,
    isLoading: query.isLoading,
    refetch: query.refetch,
  };
}

/** Admin email — hardcoded for Fight Night. */
export const ADMIN_EMAIL = "fairchild.mattie@gmail.com";

export function isAdmin(email: string | null): boolean {
  return !!email && email.toLowerCase() === ADMIN_EMAIL;
}

export interface TournamentSummary {
  id: string;
  name: string;
  status: string;
  entry_count: number;
  featured: boolean;
}

async function fetchAllTournaments(): Promise<TournamentSummary[]> {
  try {
    const res = await fetch(`${ARENA_URL}/tournaments`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.tournaments ?? [];
  } catch {
    return [];
  }
}

export function useAllTournaments(enabled: boolean) {
  const query = useQuery({
    queryKey: ["allTournaments"],
    queryFn: fetchAllTournaments,
    enabled,
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  return {
    tournaments: query.data ?? [],
    isLoading: query.isLoading,
  };
}

async function fetchTournamentById(id: string): Promise<TournamentData | null> {
  try {
    const res = await fetch(`${ARENA_URL}/tournaments/${id}/bracket`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export function useTournamentById(id: string | null) {
  const query = useQuery({
    queryKey: ["tournament", id],
    queryFn: () => fetchTournamentById(id!),
    enabled: !!id,
    refetchInterval: 5_000,
    staleTime: 3_000,
  });

  return {
    tournament: query.data ?? null,
    isLoading: query.isLoading,
  };
}

/** Find the current "playing" match in a tournament. */
export function usePlayingMatch(tournament: TournamentData | null) {
  if (!tournament?.bracket?.rounds) return null;
  for (const round of tournament.bracket.rounds) {
    for (const match of round) {
      if (match.status === "playing" && match.entry_a && match.entry_b) {
        return match;
      }
    }
  }
  return null;
}
