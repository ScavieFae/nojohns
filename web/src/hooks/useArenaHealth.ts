import { useQuery } from "@tanstack/react-query";
import { ARENA_URL } from "../config";

interface ArenaHealth {
  status: string;
  matches_played: number;
  agents_in_queue: number;
  active_matches: number;
}

export function useArenaHealth() {
  return useQuery({
    queryKey: ["arenaHealth"],
    queryFn: async (): Promise<ArenaHealth | null> => {
      try {
        const res = await fetch(`${ARENA_URL}/health`);
        if (!res.ok) return null;
        return await res.json();
      } catch {
        return null;
      }
    },
    refetchInterval: 5000,
    staleTime: 3000,
    retry: false,
  });
}
