import { useQuery } from "@tanstack/react-query";
import { wagerAbi } from "../abi/wager";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { getBatchedLogs } from "../lib/getLogs";

const wagerSettledEvent = wagerAbi.find(
  (e) => e.type === "event" && e.name === "WagerSettled",
)!;

export interface SettledWager {
  wagerId: bigint;
  matchId: `0x${string}`;
  winner: `0x${string}`;
  payout: bigint;
}

export function useSettledWagers() {
  return useQuery({
    queryKey: ["settledWagers"],
    queryFn: async (): Promise<Map<`0x${string}`, SettledWager>> => {
      if (USE_MOCK_DATA) return new Map();

      const { logs } = await getBatchedLogs({
        address: CONTRACTS.wager,
        event: wagerSettledEvent,
        cacheKey: "wagerSettled",
      });

      const byMatchId = new Map<`0x${string}`, SettledWager>();
      for (const log of logs) {
        const w: SettledWager = {
          wagerId: log.args.wagerId!,
          matchId: log.args.matchId! as `0x${string}`,
          winner: log.args.winner!,
          payout: log.args.payout!,
        };
        byMatchId.set(w.matchId, w);
      }
      return byMatchId;
    },
    refetchInterval: 30_000,
  });
}
