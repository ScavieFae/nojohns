import { useQuery } from "@tanstack/react-query";
import { publicClient } from "../viem";
import { wagerAbi } from "../abi/wager";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { MOCK_WAGERS } from "../lib/mockData";
import type { WagerRecord } from "../types";

export function useWagerEvents() {
  return useQuery({
    queryKey: ["wagerEvents"],
    queryFn: async (): Promise<WagerRecord[]> => {
      if (USE_MOCK_DATA) return MOCK_WAGERS;

      const logs = await publicClient.getLogs({
        address: CONTRACTS.wager,
        event: wagerAbi.find((e) => e.type === "event" && e.name === "WagerProposed")!,
        fromBlock: 0n,
        toBlock: "latest",
      });

      return logs.map((log) => ({
        wagerId: log.args.wagerId!,
        proposer: log.args.proposer!,
        opponent: log.args.opponent!,
        gameId: log.args.gameId!,
        amount: log.args.amount!,
        status: 0, // would need to read contract state for current status
      }));
    },
  });
}
