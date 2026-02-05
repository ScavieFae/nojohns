import { useQuery } from "@tanstack/react-query";
import { matchProofAbi } from "../abi/matchProof";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { MOCK_MATCHES } from "../lib/mockData";
import { getBatchedLogs } from "../lib/getLogs";
import type { MatchRecord } from "../types";

const matchRecordedEvent = matchProofAbi.find(
  (e) => e.type === "event" && e.name === "MatchRecorded",
)!;

export function useMatchEvents() {
  return useQuery({
    queryKey: ["matchEvents"],
    queryFn: async (): Promise<MatchRecord[]> => {
      if (USE_MOCK_DATA) return MOCK_MATCHES;

      const logs = await getBatchedLogs({
        address: CONTRACTS.matchProof,
        event: matchRecordedEvent,
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
