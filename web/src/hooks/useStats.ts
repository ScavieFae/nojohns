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

// ---------------------------------------------------------------------------
// High-water mark — stats should never decrease (partial scans, stale cache)
// ---------------------------------------------------------------------------

const HWM_KEY = "nj:stats-hwm";

interface HighWaterMark {
  totalMatches: number;
  totalWageredWei: string; // bigint as string
  predictionVolumeWei: string;
}

function loadHWM(): HighWaterMark {
  try {
    const raw = localStorage.getItem(HWM_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { totalMatches: 0, totalWageredWei: "0", predictionVolumeWei: "0" };
}

function saveHWM(hwm: HighWaterMark): void {
  try {
    localStorage.setItem(HWM_KEY, JSON.stringify(hwm));
  } catch { /* ignore */ }
}

function maxBigint(a: bigint, b: bigint): bigint {
  return a > b ? a : b;
}

// ---------------------------------------------------------------------------

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
    const hwm = loadHWM();

    const rawMatches = matches?.length ?? 0;
    const rawWagered = wagers?.reduce((sum, w) => sum + w.amount, 0n) ?? 0n;
    const rawPrediction = prediction?.totalVolume ?? 0n;

    // Apply high-water mark — never show less than we've shown before
    const totalMatches = Math.max(rawMatches, hwm.totalMatches);
    const totalWagered = maxBigint(rawWagered, BigInt(hwm.totalWageredWei));
    const predictionVolume = maxBigint(rawPrediction, BigInt(hwm.predictionVolumeWei));

    // Warn if we regressed (partial scan)
    if (rawMatches < hwm.totalMatches || rawWagered < BigInt(hwm.totalWageredWei) || rawPrediction < BigInt(hwm.predictionVolumeWei)) {
      console.warn("[useStats] Partial scan detected — using high-water mark. Raw:", {
        matches: rawMatches,
        wagered: rawWagered.toString(),
        prediction: rawPrediction.toString(),
      });
    }

    // Update high-water mark if any value increased
    if (totalMatches > hwm.totalMatches || totalWagered > BigInt(hwm.totalWageredWei) || predictionVolume > BigInt(hwm.predictionVolumeWei)) {
      saveHWM({
        totalMatches,
        totalWageredWei: totalWagered.toString(),
        predictionVolumeWei: predictionVolume.toString(),
      });
    }

    const predictionMon = predictionVolume > 0n
      ? Number(formatEther(predictionVolume)).toFixed(2)
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
