import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { formatEther } from "viem";
import { useMatchEvents } from "./useMatchEvents";
import { useWagerEvents } from "./useWagerEvents";
import { publicClient } from "../viem";
import { CONTRACTS } from "../config";
import { predictionPoolAbi } from "../abi/predictionPool";
import type { ProtocolStats } from "../types";

function usePredictionVolume() {
  return useQuery({
    queryKey: ["predictionVolume"],
    queryFn: async () => {
      const address = CONTRACTS.predictionPool as `0x${string}`;
      const balance = await publicClient.getBalance({ address });
      const poolCount = await publicClient.readContract({
        address,
        abi: predictionPoolAbi,
        functionName: "poolCount",
      }) as bigint;
      return { balance, poolCount };
    },
    staleTime: 30_000,
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

    const predictionMon = prediction?.balance
      ? Number(formatEther(prediction.balance)).toFixed(2)
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
