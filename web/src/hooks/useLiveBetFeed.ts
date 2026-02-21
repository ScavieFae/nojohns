/**
 * Hook that polls for recent prediction pool events (BetPlaced, PoolCreated, PoolResolved)
 * and maintains a rolling feed of activity.
 *
 * Uses Monad's 100-block getLogs limit — polls last 100 blocks per tick (~40s at 0.4s/block).
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { formatEther, parseAbiItem } from "viem";
import { publicClient } from "../viem";
import { CONTRACTS } from "../config";
import { truncateAddress } from "../lib/addresses";

export interface FeedEvent {
  id: string; // unique key for react
  timestamp: number; // Date.now() when we saw it
  blockNumber: bigint;
  txHash: string;
  type: "bet" | "pool_created" | "pool_resolved" | "pool_cancelled";
  text: string; // formatted display text
  poolId: number;
  // Bet-specific
  bettor?: string;
  amount?: string; // formatted MON
  side?: "A" | "B";
  // Resolve-specific
  winner?: string;
  totalPool?: string;
}

const BET_PLACED_EVENT = parseAbiItem(
  "event BetPlaced(uint256 indexed poolId, address indexed bettor, bool betOnA, uint256 amount)"
);
const POOL_CREATED_EVENT = parseAbiItem(
  "event PoolCreated(uint256 indexed poolId, bytes32 indexed matchId, address indexed playerA, address playerB)"
);
const POOL_RESOLVED_EVENT = parseAbiItem(
  "event PoolResolved(uint256 indexed poolId, address indexed winner, uint256 totalPool, uint256 fee)"
);

const MAX_FEED_SIZE = 50;
const POLL_INTERVAL_MS = 3_000;
// Monad: 100-block max per getLogs, but we scan a wider range on initial load
const BLOCKS_PER_QUERY = 100n;
const INITIAL_LOOKBACK = 2000n; // ~13 minutes at 0.4s/block

/**
 * Fetch logs in chunks of 100 blocks
 */
async function fetchLogsChunked(
  address: `0x${string}`,
  event: ReturnType<typeof parseAbiItem>,
  fromBlock: bigint,
  toBlock: bigint,
) {
  if (fromBlock > toBlock) return [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const allLogs: any[] = [];
  for (let from = fromBlock; from <= toBlock; from += BLOCKS_PER_QUERY) {
    const to = from + BLOCKS_PER_QUERY - 1n > toBlock ? toBlock : from + BLOCKS_PER_QUERY - 1n;
    try {
      const logs = await publicClient.getLogs({
        address,
        event: event as any, // eslint-disable-line @typescript-eslint/no-explicit-any
        fromBlock: from,
        toBlock: to,
      });
      allLogs.push(...logs);
    } catch {
      // Silently skip failed chunks
    }
  }
  return allLogs;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseBetEvent(log: any): FeedEvent {
  const poolId = Number(log.args.poolId);
  const bettor = truncateAddress(log.args.bettor);
  const amount = formatEther(log.args.amount);
  const side = log.args.betOnA ? "A" : "B";
  return {
    id: `${log.transactionHash}-${log.logIndex}`,
    timestamp: Date.now(),
    blockNumber: log.blockNumber,
    txHash: log.transactionHash,
    type: "bet",
    text: `bet ${amount} MON on Player ${side}`,
    poolId,
    bettor,
    amount,
    side,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parsePoolCreatedEvent(log: any): FeedEvent {
  const poolId = Number(log.args.poolId);
  const playerA = truncateAddress(log.args.playerA);
  const playerB = truncateAddress(log.args.playerB);
  return {
    id: `${log.transactionHash}-${log.logIndex}`,
    timestamp: Date.now(),
    blockNumber: log.blockNumber,
    txHash: log.transactionHash,
    type: "pool_created",
    text: `Pool opened · ${playerA} vs ${playerB}`,
    poolId,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parsePoolResolvedEvent(log: any): FeedEvent {
  const poolId = Number(log.args.poolId);
  const winner = truncateAddress(log.args.winner);
  const totalPool = formatEther(log.args.totalPool);
  return {
    id: `${log.transactionHash}-${log.logIndex}`,
    timestamp: Date.now(),
    blockNumber: log.blockNumber,
    txHash: log.transactionHash,
    type: "pool_resolved",
    text: `Pool resolved · Winner: ${winner} · ${totalPool} MON`,
    poolId,
    winner,
    totalPool,
  };
}

export function useLiveBetFeed() {
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const lastBlockRef = useRef<bigint>(0n);
  const seenIdsRef = useRef<Set<string>>(new Set());

  const addEvents = useCallback((newEvents: FeedEvent[]) => {
    setEvents((prev) => {
      const unseen = newEvents.filter((e) => !seenIdsRef.current.has(e.id));
      if (unseen.length === 0) return prev;
      for (const e of unseen) seenIdsRef.current.add(e.id);
      const merged = [...prev, ...unseen]
        .sort((a, b) => Number(a.blockNumber - b.blockNumber))
        .slice(-MAX_FEED_SIZE);
      return merged;
    });
  }, []);

  const poll = useCallback(async (fromBlock: bigint, toBlock: bigint) => {
    const address = CONTRACTS.predictionPool as `0x${string}`;
    if (address === "0x0000000000000000000000000000000000000000") return;

    const [bets, created, resolved] = await Promise.all([
      fetchLogsChunked(address, BET_PLACED_EVENT, fromBlock, toBlock),
      fetchLogsChunked(address, POOL_CREATED_EVENT, fromBlock, toBlock),
      fetchLogsChunked(address, POOL_RESOLVED_EVENT, fromBlock, toBlock),
    ]);

    const parsed: FeedEvent[] = [
      ...bets.map(parseBetEvent),
      ...created.map(parsePoolCreatedEvent),
      ...resolved.map(parsePoolResolvedEvent),
    ];

    addEvents(parsed);
    lastBlockRef.current = toBlock;
  }, [addEvents]);

  // Initial load — scan recent history
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const latest = await publicClient.getBlockNumber();
        const from = latest > INITIAL_LOOKBACK ? latest - INITIAL_LOOKBACK : 0n;
        if (!cancelled) {
          await poll(from, latest);
          setIsLoading(false);
        }
      } catch {
        if (!cancelled) setIsLoading(false);
      }
    }

    init();
    return () => { cancelled = true; };
  }, [poll]);

  // Polling loop
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const latest = await publicClient.getBlockNumber();
        const from = lastBlockRef.current > 0n ? lastBlockRef.current + 1n : latest - BLOCKS_PER_QUERY;
        if (from <= latest) {
          await poll(from, latest);
        }
      } catch {
        // Silently retry next tick
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [poll]);

  return { events, isLoading };
}
