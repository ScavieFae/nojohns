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
        {match.transactionHash && (
          <a
            href={explorerLink("tx", match.transactionHash)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-blue text-xs font-mono hover:underline"
          >
            proof
          </a>
        )}
      </td>
    </tr>
  );
}
