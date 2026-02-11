import { type AbiEvent, decodeEventLog } from "viem";
import { publicClient } from "../viem";
import { DEPLOY_BLOCK, MAX_LOG_RANGE } from "../config";
import { SNAPSHOT } from "./snapshot";

/** Max concurrent RPC requests (stay under Monad rate limits) */
const CONCURRENCY = 50;
/** Retry failed requests up to this many times */
const MAX_RETRIES = 3;
/** Base delay for exponential backoff (ms) */
const RETRY_BASE_MS = 1000;
/** Max time to spend scanning per call — resumes on next call */
const SCAN_BUDGET_MS = 30_000;

// ---------------------------------------------------------------------------
// localStorage cache with bigint serialization
// ---------------------------------------------------------------------------

const BI_PREFIX = "\0bi:";

function serialize(data: unknown): string {
  return JSON.stringify(data, (_, v) =>
    typeof v === "bigint" ? BI_PREFIX + v.toString() : v,
  );
}

function deserialize(json: string): unknown {
  return JSON.parse(json, (_, v) =>
    typeof v === "string" && v.startsWith(BI_PREFIX)
      ? BigInt(v.slice(BI_PREFIX.length))
      : v,
  );
}

interface LogCacheEntry {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  logs: any[];
  scannedToBlock: bigint;
}

function loadCache(key: string): LogCacheEntry | null {
  try {
    const raw = localStorage.getItem(`nj:${key}`);
    if (!raw) return null;
    return deserialize(raw) as LogCacheEntry;
  } catch {
    return null;
  }
}

function saveCache(key: string, entry: LogCacheEntry): void {
  try {
    localStorage.setItem(`nj:${key}`, serialize(entry));
  } catch {
    // localStorage full — clear our entries and retry
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const k = localStorage.key(i);
      if (k?.startsWith("nj:")) localStorage.removeItem(k);
    }
    try {
      localStorage.setItem(`nj:${key}`, serialize(entry));
    } catch {
      // give up silently
    }
  }
}

// ---------------------------------------------------------------------------
// Snapshot decoding — raw RPC logs → viem-style logs with decoded args
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function decodeSnapshotLogs(logs: any[], event: AbiEvent): any[] {
  return logs.map((log) => {
    try {
      const { args } = decodeEventLog({
        abi: [event],
        data: log.data,
        topics: log.topics,
        strict: false,
      });
      return { ...log, args };
    } catch {
      return log;
    }
  });
}

// ---------------------------------------------------------------------------
// Parallel scanning with retry
// ---------------------------------------------------------------------------

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchWithRetry(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fn: () => Promise<any[]>,
  retries = MAX_RETRIES,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Promise<any[]> {
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch {
      if (i === retries) return []; // fail silently on last attempt
      await sleep(RETRY_BASE_MS * Math.pow(2, i));
    }
  }
  return [];
}

/**
 * Batched getLogs with parallel scanning, retry, time budget, and localStorage cache.
 *
 * - Fires CONCURRENCY requests at a time to stay under Monad rate limits
 * - Retries failed requests with exponential backoff
 * - Caches results + scan progress in localStorage (survives page refresh)
 * - Time-budgeted: won't scan for more than SCAN_BUDGET_MS per call,
 *   and resumes on the next call from where it left off
 *
 * @param cacheKey  If provided, enables localStorage persistence. Use a unique
 *                  key per contract+event combination.
 */
export async function getBatchedLogs({
  address,
  event,
  fromBlock,
  cacheKey,
}: {
  address: `0x${string}`;
  event: AbiEvent;
  fromBlock?: bigint;
  cacheKey?: string;
}) {
  // 1. Load cached data (localStorage > bundled snapshot > empty)
  //    Snapshot logs are raw RPC format — decode them via decodeEventLog on first use.
  //    Once decoded and saved to localStorage, future loads skip decoding.
  const cached = cacheKey ? loadCache(cacheKey) : null;
  const snapshotData = cacheKey ? SNAPSHOT[cacheKey] ?? null : null;
  const needsDecode = !cached && snapshotData;
  const seed = cached ?? snapshotData;
  const cachedLogs = needsDecode
    ? decodeSnapshotLogs(snapshotData!.logs, event)
    : seed?.logs ?? [];

  // 2. Determine start block (explicit fromBlock > cache/snapshot > deploy block)
  const startBlock =
    fromBlock ??
    (seed?.scannedToBlock != null ? seed.scannedToBlock + 1n : DEPLOY_BLOCK);

  // 3. Get current block
  const latest = await publicClient.getBlockNumber();
  if (startBlock > latest) {
    return { logs: cachedLogs, scannedToBlock: latest };
  }

  // 4. Build batch ranges
  const ranges: { from: bigint; to: bigint }[] = [];
  for (let from = startBlock; from <= latest; from += MAX_LOG_RANGE) {
    const to =
      from + MAX_LOG_RANGE - 1n > latest ? latest : from + MAX_LOG_RANGE - 1n;
    ranges.push({ from, to });
  }

  // 5. Scan in parallel with time budget
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const newLogs: any[] = [];
  const t0 = Date.now();
  let lastScannedTo = startBlock - 1n;

  for (let i = 0; i < ranges.length; i += CONCURRENCY) {
    if (Date.now() - t0 > SCAN_BUDGET_MS) break;

    const batch = ranges.slice(i, i + CONCURRENCY);
    const results = await Promise.all(
      batch.map(({ from, to }) =>
        fetchWithRetry(() =>
          publicClient.getLogs({ address, event, fromBlock: from, toBlock: to }),
        ),
      ),
    );

    for (const result of results) {
      newLogs.push(...result);
    }
    lastScannedTo = batch[batch.length - 1].to;
  }

  const allLogs = [...cachedLogs, ...newLogs];
  const scannedToBlock =
    lastScannedTo >= startBlock
      ? lastScannedTo
      : cached?.scannedToBlock ?? DEPLOY_BLOCK;

  // 6. Persist cache
  if (cacheKey) {
    saveCache(cacheKey, { logs: allLogs, scannedToBlock });
  }

  return { logs: allLogs, scannedToBlock };
}
