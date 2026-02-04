import { useQuery } from "@tanstack/react-query";
import { publicClient } from "../viem";
import { matchProofAbi } from "../abi/matchProof";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { MOCK_MATCHES } from "../lib/mockData";
import type { MatchRecord } from "../types";

export function useMatchEvents() {
  return useQuery({
    queryKey: ["matchEvents"],
    queryFn: async (): Promise<MatchRecord[]> => {
      if (USE_MOCK_DATA) return MOCK_MATCHES;

      const logs = await publicClient.getLogs({
        address: CONTRACTS.matchProof,
        event: matchProofAbi.find((e) => e.type === "event" && e.name === "MatchRecorded")!,
        fromBlock: 0n,
        toBlock: "latest",
      });

      return logs.map((log) => ({
        matchId: log.args.matchId!,
        winner: log.args.winner!,
        loser: log.args.loser!,
        gameId: log.args.gameId!,
        winnerScore: log.args.winnerScore!,
        loserScore: log.args.loserScore!,
        replayHash: log.args.replayHash!,
        timestamp: log.args.timestamp!,
        blockNumber: log.blockNumber,
        transactionHash: log.transactionHash,
      }));
    },
  });
}
