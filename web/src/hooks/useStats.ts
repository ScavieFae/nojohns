import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { formatEther } from "viem";
import { useMatchEvents } from "./useMatchEvents";
import { useWagerEvents } from "./useWagerEvents";
import { CONTRACTS } from "../config";
import { predictionPoolAbi } from "../abi/predictionPool";
import { getBatchedLogs } from "../lib/getLogs";
import type { ProtocolStats } from "../types";

const betPlacedEvent = predictionPoolAbi.find(
  (e) => e.type === "event" && e.name === "BetPlaced",
)!;

function usePredictionVolume() {
  return useQuery({
    queryKey: ["predictionVolume"],
    queryFn: async () => {
      const { logs } = await getBatchedLogs({
        address: CONTRACTS.predictionPool,
        event: betPlacedEvent,
        cacheKey: "betPlaced",
      });
      const totalVolume = logs.reduce(
        (sum: bigint, log: Record<string, any>) => sum + (log.args.amount ?? 0n),
        0n,
      );
      return { totalVolume, betCount: logs.length };
    },
    refetchInterval: 30_000,
  });
}

export function useStats(): ProtocolStats & { isLoading: boolean; isError: boolean } {
  const { data: matches, isLoading: matchesLoading, isError: matchesError } = useMatchEvents();
  const { data: wagers, isLoading: wagersLoading, isError: wagersError } = useWagerEvents();
  const { data: prediction, isLoading: predictionLoading } = usePredictionVolume();

  const stats = useMemo(() => {
    const totalMatches = matches?.length ?? 0;

    const totalWagered = wagers?.reduce((sum, w) => sum + w.amount, 0n) ?? 0n;

    const predictionMon = prediction?.totalVolume
      ? Number(formatEther(prediction.totalVolume)).toFixed(2)
      : "0.00";

    return {
      totalMatches,
      totalWagered,
      uniqueAgents: 0,
      predictionVolume: predictionMon,
    };
  }, [matches, wagers, prediction]);

  return { ...stats, isLoading: matchesLoading || wagersLoading || predictionLoading, isError: matchesError || wagersError };
}
