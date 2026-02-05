import { AddressDisplay } from "../shared/AddressDisplay";
import { ScoreDisplay } from "../shared/ScoreDisplay";
import { explorerLink } from "../../lib/addresses";
import type { MatchRecord } from "../../types";

interface MatchRowProps {
  match: MatchRecord;
}

function formatTimestamp(timestamp: bigint): string {
  const date = new Date(Number(timestamp) * 1000);
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));

  if (diffHours < 1) return "< 1h ago";
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function ProofLink({ hash }: { hash?: string }) {
  if (!hash) return null;
  return (
    <a
      href={explorerLink("tx", hash)}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent-blue text-xs font-mono hover:underline"
    >
      proof
    </a>
  );
}

export function MatchRow({ match }: MatchRowProps) {
  return (
    <tr className="border-b border-surface-700 hover:bg-surface-700/50 transition-colors">
      <td className="py-3 px-4 text-sm text-gray-500">
        {formatTimestamp(match.timestamp)}
      </td>
      <td className="py-3 px-4">
        <AddressDisplay address={match.winner} />
      </td>
      <td className="py-3 px-4 text-center">
        <ScoreDisplay winnerScore={match.winnerScore} loserScore={match.loserScore} />
      </td>
      <td className="py-3 px-4">
        <AddressDisplay address={match.loser} />
      </td>
      <td className="py-3 px-4">
        <span className="bg-surface-700 text-gray-400 text-xs font-mono px-2 py-1 rounded">
          {match.gameId}
        </span>
      </td>
      <td className="py-3 px-4 text-right">
        <ProofLink hash={match.transactionHash} />
      </td>
    </tr>
  );
}

export function MatchCard({ match }: MatchRowProps) {
  return (
    <div className="border-b border-surface-700 p-4 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-500">{formatTimestamp(match.timestamp)}</span>
        <div className="flex items-center gap-2">
          <span className="bg-surface-700 text-gray-400 text-xs font-mono px-2 py-1 rounded">
            {match.gameId}
          </span>
          <ProofLink hash={match.transactionHash} />
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <div className="text-xs text-gray-500 mb-0.5">Winner</div>
          <AddressDisplay address={match.winner} />
        </div>
        <ScoreDisplay winnerScore={match.winnerScore} loserScore={match.loserScore} />
        <div className="flex-1 min-w-0 text-right">
          <div className="text-xs text-gray-500 mb-0.5">Loser</div>
          <AddressDisplay address={match.loser} />
        </div>
      </div>
    </div>
  );
}
