/**
 * Pre-scanned event snapshot — seeds localStorage cache on first visit.
 *
 * Without this, the site must scan 1.4M+ blocks at 100 blocks per RPC call.
 * With it, only blocks AFTER the snapshot need scanning (seconds, not minutes).
 *
 * Generated: 2026-02-11 (block 54718536)
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

export const SNAPSHOT: Record<string, SnapshotData> = {
  matchRecorded: {
    scannedToBlock: 54718536n,
    logs: [
    ],
  },
  wagerProposed: {
    scannedToBlock: 54718536n,
    logs: [
    ],
  },
  wagerSettled: {
    scannedToBlock: 54718536n,
    logs: [
    ],
  },
  eloFeedback: {
    scannedToBlock: 54718536n,
    logs: [
    ],
  },
};
