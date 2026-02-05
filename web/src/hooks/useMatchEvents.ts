import { useQuery, useQueryClient } from "@tanstack/react-query";
import { matchProofAbi } from "../abi/matchProof";
import { CONTRACTS, USE_MOCK_DATA } from "../config";
import { MOCK_MATCHES } from "../lib/mockData";
import { getBatchedLogs } from "../lib/getLogs";
import type { MatchRecord } from "../types";

const matchRecordedEvent = matchProofAbi.find(
  (e) => e.type === "event" && e.name === "MatchRecorded",
)!;

interface CachedMatchData {
  matches: MatchRecord[];
  scannedToBlock: bigint;
}

export function useMatchEvents() {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ["matchEvents"],
    queryFn: async (): Promise<MatchRecord[]> => {
      if (USE_MOCK_DATA) return MOCK_MATCHES;

      // Check for previously cached data to do incremental scanning
      const prev = queryClient.getQueryData<MatchRecord[]>(["matchEvents"]);
      const prevMeta = queryClient.getQueryData<CachedMatchData>(["matchEvents_meta"]);

      // If we have previous results, only scan from where we left off
      const fromBlock = prevMeta?.scannedToBlock ? prevMeta.scannedToBlock + 1n : undefined;

      const { logs, scannedToBlock } = await getBatchedLogs({
        address: CONTRACTS.matchProof,
        event: matchRecordedEvent,
        fromBlock,
      });

      const newMatches = logs.map((log) => ({
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

      const allMatches = prev && fromBlock ? [...prev, ...newMatches] : newMatches;

      // Store metadata for next incremental fetch
      queryClient.setQueryData<CachedMatchData>(["matchEvents_meta"], {
        matches: allMatches,
        scannedToBlock,
      });

      return allMatches;
    },
  });
}
