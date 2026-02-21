#!/usr/bin/env node
/**
 * Regenerate web/src/lib/snapshot.ts with current on-chain events.
 *
 * Usage:  node web/scripts/generate-snapshot.mjs
 *
 * Scans from DEPLOY_BLOCK to the current block, fetching all relevant events
 * from the MatchProof, Wager, and ReputationRegistry contracts. Respects the
 * Monad testnet 100-block getLogs limit.
 */

// Defaults to mainnet; pass --testnet for testnet
const isTestnet = process.argv.includes("--testnet");
const RPC = isTestnet
  ? "https://testnet-rpc.monad.xyz"
  : "https://rpc.monad.xyz";
const DEPLOY_BLOCK = isTestnet ? 10710000 : 54717354;
const MAX_RANGE = 100; // Monad caps getLogs at 100 blocks per request
const CONCURRENCY = 10; // Lower to avoid Monad rate limits (was 50, caused silent drops)
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 500;

const EVENTS = {
  matchRecorded: {
    address: "0x1CC748475F1F666017771FB49131708446B9f3DF",
    topic0:
      "0x85577f91f88e7f39d7e011ec5acc580d01b464a746a050601a5ac73042f43566",
  },
  wagerProposed: {
    address: "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d",
    topic0:
      "0x1f4b3c2984c1abae822b24ee17eaf2f1dae5338c5da04a753d15a6af1c94f365",
  },
  wagerSettled: {
    address: "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d",
    topic0:
      "0x17a11cc71694f13ce59da349be2384455ee1b564012052f6b25b315fa42dabb1",
  },
  betPlaced: {
    address: "0x33E65E300575D11a42a579B2675A63cb4374598D",
    topic0:
      "0x4af71b021e799c62c158bd54636ca8da2fa26115a21a2dc6efe486ec104fd15f",
  },
  eloFeedback: {
    address: isTestnet
      ? "0x8004B663056A597Dffe9eCcC1965A193B7388713"
      : "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63",
    topic0:
      "0xf5c8dfad1418627c565486bc917258f166ee28e7d5c6a387b6fade859a4ba51a",
  },
};

// ---------------------------------------------------------------------------
// JSON-RPC helpers
// ---------------------------------------------------------------------------

let rpcId = 0;

async function rpc(method, params) {
  const res = await fetch(RPC, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: ++rpcId, method, params }),
  });
  const json = await res.json();
  if (json.error) throw new Error(`RPC error: ${json.error.message}`);
  return json.result;
}

function hex(n) {
  return "0x" + n.toString(16);
}

async function getBlockNumber() {
  const result = await rpc("eth_blockNumber", []);
  return parseInt(result, 16);
}

async function getLogs(address, topic0, from, to) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await rpc("eth_getLogs", [
        {
          address,
          topics: [topic0],
          fromBlock: hex(from),
          toBlock: hex(to),
        },
      ]);
    } catch (err) {
      if (attempt === MAX_RETRIES) {
        process.stderr.write(`\n  WARN: getLogs failed after ${MAX_RETRIES} retries (${hex(from)}-${hex(to)}): ${err.message}\n`);
        return [];
      }
      await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * (attempt + 1)));
    }
  }
}

// ---------------------------------------------------------------------------
// Parallel scanner
// ---------------------------------------------------------------------------

async function scanEvent(name, { address, topic0 }, latest) {
  const ranges = [];
  for (let from = DEPLOY_BLOCK; from <= latest; from += MAX_RANGE) {
    const to = Math.min(from + MAX_RANGE - 1, latest);
    ranges.push({ from, to });
  }

  const allLogs = [];
  const total = ranges.length;
  let done = 0;

  for (let i = 0; i < ranges.length; i += CONCURRENCY) {
    const batch = ranges.slice(i, i + CONCURRENCY);
    const results = await Promise.all(
      batch.map(({ from, to }) =>
        getLogs(address, topic0, from, to),
      ),
    );
    for (const logs of results) {
      if (Array.isArray(logs)) allLogs.push(...logs);
    }
    done += batch.length;
    process.stderr.write(
      `\r  ${name}: ${done}/${total} ranges (${allLogs.length} events)`,
    );
  }
  process.stderr.write("\n");
  return allLogs;
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

function formatLog(log) {
  // Convert hex blockNumber to bigint literal for the TS file
  const blockNum = parseInt(log.blockNumber, 16);
  const logIdx = parseInt(log.logIndex, 16);
  const txIdx = parseInt(log.transactionIndex, 16);

  return `    {
      address: ${JSON.stringify(log.address)},
      blockNumber: ${blockNum}n,
      data: ${JSON.stringify(log.data)},
      logIndex: ${logIdx},
      transactionHash: ${JSON.stringify(log.transactionHash)},
      transactionIndex: ${txIdx},
      blockHash: ${JSON.stringify(log.blockHash)},
      topics: ${JSON.stringify(log.topics)},
    }`;
}

function generateTs(eventLogs, scannedToBlock) {
  const date = new Date().toISOString().slice(0, 10);

  let ts = `/**
 * Pre-scanned event snapshot — seeds localStorage cache on first visit.
 *
 * Without this, the site must scan 1.4M+ blocks at 100 blocks per RPC call.
 * With it, only blocks AFTER the snapshot need scanning (seconds, not minutes).
 *
 * Generated: ${date} (block ${scannedToBlock})
 *
 * Regenerate: node web/scripts/generate-snapshot.mjs
 */

// Raw RPC log entries — getBatchedLogs decodes them via decodeEventLog
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RawLog = Record<string, any>;

interface SnapshotData {
  logs: RawLog[];
  scannedToBlock: bigint;
}

export const SNAPSHOT: Record<string, SnapshotData> = {\n`;

  for (const [key, logs] of Object.entries(eventLogs)) {
    ts += `  ${key}: {\n`;
    ts += `    scannedToBlock: ${scannedToBlock}n,\n`;
    ts += `    logs: [\n`;
    ts += logs.map(formatLog).join(",\n");
    if (logs.length > 0) ts += ",\n";
    ts += `    ],\n`;
    ts += `  },\n`;
  }

  ts += `};\n`;
  return ts;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.error("Fetching current block...");
  const latest = await getBlockNumber();
  console.error(`Current block: ${latest}`);
  console.error(
    `Scanning ${latest - DEPLOY_BLOCK} blocks (${Math.ceil((latest - DEPLOY_BLOCK) / MAX_RANGE)} ranges per event)\n`,
  );

  const eventLogs = {};
  for (const [name, config] of Object.entries(EVENTS)) {
    eventLogs[name] = await scanEvent(name, config, latest);
  }

  const ts = generateTs(eventLogs, latest);

  const path = new URL("../src/lib/snapshot.ts", import.meta.url);
  const { writeFileSync } = await import("fs");
  const { fileURLToPath } = await import("url");
  writeFileSync(fileURLToPath(path), ts);

  console.error(`\nSnapshot written to web/src/lib/snapshot.ts`);
  for (const [name, logs] of Object.entries(eventLogs)) {
    console.error(`  ${name}: ${logs.length} events`);
  }
  console.error(`  scannedToBlock: ${latest}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
