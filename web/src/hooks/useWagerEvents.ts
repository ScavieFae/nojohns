import { useQuery, useQueryClient } from "@tanstack/react-query";
import { wagerAbi } from "../abi/wager";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { MOCK_WAGERS } from "../lib/mockData";
import { getBatchedLogs } from "../lib/getLogs";
import type { WagerRecord } from "../types";

const wagerProposedEvent = wagerAbi.find(
  (e) => e.type === "event" && e.name === "WagerProposed",
)!;

interface CachedWagerData {
  wagers: WagerRecord[];
  scannedToBlock: bigint;
}

export function useWagerEvents() {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ["wagerEvents"],
    queryFn: async (): Promise<WagerRecord[]> => {
      if (USE_MOCK_DATA) return MOCK_WAGERS;

      const prev = queryClient.getQueryData<WagerRecord[]>(["wagerEvents"]);
      const prevMeta = queryClient.getQueryData<CachedWagerData>(["wagerEvents_meta"]);

      const fromBlock = prevMeta?.scannedToBlock ? prevMeta.scannedToBlock + 1n : undefined;

      const { logs, scannedToBlock } = await getBatchedLogs({
        address: CONTRACTS.wager,
        event: wagerProposedEvent,
        fromBlock,
      });

      const newWagers = logs.map((log) => ({
        wagerId: log.args.wagerId!,
        proposer: log.args.proposer!,
        opponent: log.args.opponent!,
        gameId: log.args.gameId!,
        amount: log.args.amount!,
        status: 0, // would need to read contract state for current status
      }));

      const allWagers = prev && fromBlock ? [...prev, ...newWagers] : newWagers;

      queryClient.setQueryData<CachedWagerData>(["wagerEvents_meta"], {
        wagers: allWagers,
        scannedToBlock,
      });

      return allWagers;
    },
  });
}
