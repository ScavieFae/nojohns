import { useLeaderboard } from "../../hooks/useLeaderboard";
import { LeaderboardRow, LeaderboardCard } from "./LeaderboardRow";
import { EmptyState } from "../shared/EmptyState";

export function LeaderboardTable() {
  const { data: leaderboard, isLoading, isError, refetch } = useLeaderboard();

  if (isError) {
    return (
      <div className="bg-accent-red/10 border border-accent-red/30 rounded-lg px-4 py-3 flex items-center justify-between">
        <span className="text-accent-red text-sm">Failed to load leaderboard</span>
        <button
          onClick={() => refetch()}
          className="text-accent-red text-sm font-mono hover:underline"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-surface-700 rounded h-12 animate-pulse" />
        ))}
      </div>
    );
  }

  if (!leaderboard?.length) {
    return (
      <EmptyState
        title="No matches recorded yet"
        description="Once agents start competing, the leaderboard will populate from onchain match results."
      />
    );
  }

  return (
    <div className="bg-surface-800 border border-surface-600 rounded-lg overflow-hidden">
      {/* Desktop: table */}
      <table className="hidden md:table w-full">
        <thead>
          <tr className="border-b border-surface-600 text-left text-xs text-gray-500 uppercase tracking-wider">
            <th className="py-3 px-4 w-16">Rank</th>
            <th className="py-3 px-4">Agent</th>
            <th className="py-3 px-4">Elo</th>
            <th className="py-3 px-4">Record</th>
            <th className="py-3 px-4">Games</th>
            <th className="py-3 px-4">Win %</th>
          </tr>
        </thead>
        <tbody>
          {leaderboard.map((agent, i) => (
            <LeaderboardRow key={agent.address} agent={agent} rank={i + 1} />
          ))}
        </tbody>
      </table>

      {/* Mobile: card list */}
      <div className="md:hidden">
        {leaderboard.map((agent, i) => (
          <LeaderboardCard key={agent.address} agent={agent} rank={i + 1} />
        ))}
      </div>
    </div>
  );
}
