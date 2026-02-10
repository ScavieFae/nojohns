import { useQuery } from "@tanstack/react-query";
import { wagerAbi } from "../abi/wager";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { MOCK_WAGERS } from "../lib/mockData";
import { getBatchedLogs } from "../lib/getLogs";
import type { WagerRecord } from "../types";

const wagerProposedEvent = wagerAbi.find(
  (e) => e.type === "event" && e.name === "WagerProposed",
)!;

export function useWagerEvents() {
  return useQuery({
    queryKey: ["wagerEvents"],
    queryFn: async (): Promise<WagerRecord[]> => {
      if (USE_MOCK_DATA) return MOCK_WAGERS;

      const { logs } = await getBatchedLogs({
        address: CONTRACTS.wager,
        event: wagerProposedEvent,
        cacheKey: "wagerProposed",
      });

      return logs.map((log: Record<string, any>) => ({
        wagerId: log.args.wagerId!,
        proposer: log.args.proposer!,
        opponent: log.args.opponent!,
        gameId: log.args.gameId!,
        amount: log.args.amount!,
        status: 0,
      }));
    },
    refetchInterval: 30_000,
  });
}
