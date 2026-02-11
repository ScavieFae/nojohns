import { useQuery } from "@tanstack/react-query";
import { publicClient } from "../viem";
import { CONTRACTS } from "../config";
import { predictionPoolAbi } from "../abi/predictionPool";

export enum PoolStatus {
  Open = 0,
  Resolved = 1,
  Cancelled = 2,
}

export interface PoolState {
  poolId: bigint;
  matchId: `0x${string}`;
  playerA: `0x${string}`;
  playerB: `0x${string}`;
  totalA: bigint;
  totalB: bigint;
  status: PoolStatus;
  winner: `0x${string}`;
  createdAt: bigint;
}

export interface UserPosition {
  betOnA: bigint;
  betOnB: bigint;
  claimable: bigint;
}

/**
 * Find a prediction pool for a matchId by scanning PoolCreated events.
 * Returns null if no pool exists for this match.
 */
async function findPoolIdForMatch(
  matchId: `0x${string}`
): Promise<bigint | null> {
  const address = CONTRACTS.predictionPool as `0x${string}`;
  if (address === "0x0000000000000000000000000000000000000000") return null;

  // Check pool count first
  const count = (await publicClient.readContract({
    address,
    abi: predictionPoolAbi,
    functionName: "poolCount",
  })) as bigint;

  // Scan pools (they're sequential IDs, so just check each)
  for (let i = 0n; i < count; i++) {
    const pool = (await publicClient.readContract({
      address,
      abi: predictionPoolAbi,
      functionName: "getPool",
      args: [i],
    })) as {
      matchId: `0x${string}`;
    };
    if (pool.matchId === matchId) return i;
  }
  return null;
}

async function fetchPoolState(poolId: bigint): Promise<PoolState | null> {
  const address = CONTRACTS.predictionPool as `0x${string}`;
  if (address === "0x0000000000000000000000000000000000000000") return null;

  const result = (await publicClient.readContract({
    address,
    abi: predictionPoolAbi,
    functionName: "getPool",
    args: [poolId],
  })) as {
    matchId: `0x${string}`;
    playerA: `0x${string}`;
    playerB: `0x${string}`;
    totalA: bigint;
    totalB: bigint;
    status: number;
    winner: `0x${string}`;
    createdAt: bigint;
  };

  if (result.playerA === "0x0000000000000000000000000000000000000000") return null;

  return {
    poolId,
    matchId: result.matchId,
    playerA: result.playerA,
    playerB: result.playerB,
    totalA: result.totalA,
    totalB: result.totalB,
    status: result.status as PoolStatus,
    winner: result.winner,
    createdAt: result.createdAt,
  };
}

async function fetchUserPosition(
  poolId: bigint,
  user: `0x${string}`
): Promise<UserPosition> {
  const address = CONTRACTS.predictionPool as `0x${string}`;

  const [bets, claimable] = await Promise.all([
    publicClient.readContract({
      address,
      abi: predictionPoolAbi,
      functionName: "getUserBets",
      args: [poolId, user],
    }) as Promise<[bigint, bigint]>,
    publicClient.readContract({
      address,
      abi: predictionPoolAbi,
      functionName: "getClaimable",
      args: [poolId, user],
    }) as Promise<bigint>,
  ]);

  return { betOnA: bets[0], betOnB: bets[1], claimable };
}

/**
 * Hook to read prediction pool state for a given matchId.
 * Polls every 5s when pool is open to track odds changes.
 */
export function usePredictionPool(matchId: string | undefined) {
  // Arena match IDs are UUIDs like "5b26172f-ec93-4e76-b41e-b161ff13a7b6"
  // On-chain they're bytes32: strip dashes, right-pad with zeros
  const matchIdHex = matchId
    ? (`0x${matchId.replace(/^0x/, "").replace(/-/g, "").padEnd(64, "0")}` as `0x${string}`)
    : undefined;

  const poolIdQuery = useQuery({
    queryKey: ["predictionPoolId", matchIdHex],
    queryFn: () => findPoolIdForMatch(matchIdHex!),
    enabled: !!matchIdHex,
    staleTime: 60_000,
  });

  const poolId = poolIdQuery.data ?? undefined;

  const poolQuery = useQuery({
    queryKey: ["predictionPool", poolId?.toString()],
    queryFn: () => fetchPoolState(poolId!),
    enabled: poolId !== undefined,
    refetchInterval: 5_000, // Poll every 5s for odds updates
    staleTime: 3_000,
  });

  return {
    poolId,
    pool: poolQuery.data ?? null,
    isLoading: poolIdQuery.isLoading || poolQuery.isLoading,
    hasPool: poolId !== undefined,
  };
}

/**
 * Hook to read the user's position in a prediction pool.
 */
export function useUserPosition(
  poolId: bigint | undefined,
  account: `0x${string}` | null
) {
  const query = useQuery({
    queryKey: ["predictionPosition", poolId?.toString(), account],
    queryFn: () => fetchUserPosition(poolId!, account!),
    enabled: poolId !== undefined && !!account,
    refetchInterval: 5_000,
    staleTime: 3_000,
  });

  return query.data ?? null;
}
