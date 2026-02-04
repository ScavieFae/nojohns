import { useState, useMemo } from "react";
import { useMatchEvents } from "../../hooks/useMatchEvents";
import { MatchRow } from "./MatchRow";
import { MatchFilter } from "./MatchFilter";
import { EmptyState } from "../shared/EmptyState";

export function MatchList() {
  const { data: matches, isLoading } = useMatchEvents();
  const [filterAddress, setFilterAddress] = useState("");

  const filtered = useMemo(() => {
    if (!matches) return [];
    const sorted = [...matches].sort((a, b) => Number(b.timestamp - a.timestamp));
    if (!filterAddress) return sorted;
    const lower = filterAddress.toLowerCase();
    return sorted.filter(
      (m) =>
        m.winner.toLowerCase().includes(lower) || m.loser.toLowerCase().includes(lower),
    );
  }, [matches, filterAddress]);

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="bg-surface-700 rounded h-12 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <MatchFilter filterAddress={filterAddress} onFilterChange={setFilterAddress} />
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          title={filterAddress ? "No matches found" : "No matches recorded yet"}
          description={
            filterAddress
              ? "No matches found for this address. Try a different filter."
              : "Once agents start competing, match history will appear here from onchain events."
          }
        />
      ) : (
        <div className="bg-surface-800 border border-surface-600 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-surface-600 text-left text-xs text-gray-500 uppercase tracking-wider">
                <th className="py-3 px-4">When</th>
                <th className="py-3 px-4">Winner</th>
                <th className="py-3 px-4 text-center">Score</th>
                <th className="py-3 px-4">Loser</th>
                <th className="py-3 px-4">Game</th>
                <th className="py-3 px-4 text-right">Proof</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((match) => (
                <MatchRow key={match.matchId} match={match} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
