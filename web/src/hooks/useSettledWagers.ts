import { useQuery, useQueryClient } from "@tanstack/react-query";
import { wagerAbi } from "../abi/wager";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { getBatchedLogs } from "../lib/getLogs";

const wagerSettledEvent = wagerAbi.find(
  (e) => e.type === "event" && e.name === "WagerSettled"
)!;

export interface SettledWager {
  wagerId: bigint;
  matchId: `0x${string}`;
  winner: `0x${string}`;
  payout: bigint; // Total pot (both stakes)
}

interface CachedData {
  wagers: SettledWager[];
  scannedToBlock: bigint;
}

/**
 * Fetch WagerSettled events — includes matchId for correlation with matches
 */
export function useSettledWagers() {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ["settledWagers"],
    queryFn: async (): Promise<Map<`0x${string}`, SettledWager>> => {
      if (USE_MOCK_DATA) return new Map();

      const prev = queryClient.getQueryData<SettledWager[]>(["settledWagers"]);
      const prevMeta = queryClient.getQueryData<CachedData>(["settledWagers_meta"]);

      const fromBlock = prevMeta?.scannedToBlock ? prevMeta.scannedToBlock + 1n : undefined;

      const { logs, scannedToBlock } = await getBatchedLogs({
        address: CONTRACTS.wager,
        event: wagerSettledEvent,
        fromBlock,
      });

      const newWagers = logs.map((log) => ({
        wagerId: log.args.wagerId!,
        matchId: log.args.matchId! as `0x${string}`,
        winner: log.args.winner!,
        payout: log.args.payout!,
      }));

      const allWagers = prev && fromBlock ? [...prev, ...newWagers] : newWagers;

      queryClient.setQueryData<CachedData>(["settledWagers_meta"], {
        wagers: allWagers,
        scannedToBlock,
      });

      // Return as map keyed by matchId for easy lookup
      const byMatchId = new Map<`0x${string}`, SettledWager>();
      for (const w of allWagers) {
        byMatchId.set(w.matchId, w);
      }
      return byMatchId;
    },
  });
}
