import type { AbiEvent } from "viem";
import { publicClient } from "../viem";
import { DEPLOY_BLOCK, MAX_LOG_RANGE } from "../config";

/**
 * Batched getLogs — Monad testnet limits to 100 blocks per request.
 * Scans from `fromBlock` (default: DEPLOY_BLOCK) to latest in chunks.
 */
export async function getBatchedLogs({
  address,
  event,
  fromBlock,
}: {
  address: `0x${string}`;
  event: AbiEvent;
  fromBlock?: bigint;
}) {
  const startBlock = fromBlock ?? DEPLOY_BLOCK;
  const latest = await publicClient.getBlockNumber();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const logs: any[] = [];

  for (let from = startBlock; from <= latest; from += MAX_LOG_RANGE) {
    const to = from + MAX_LOG_RANGE - 1n > latest ? latest : from + MAX_LOG_RANGE - 1n;
    const batch = await publicClient.getLogs({
      address,
      event,
      fromBlock: from,
      toBlock: to,
    });
    logs.push(...batch);
  }

  return { logs, scannedToBlock: latest };
}
