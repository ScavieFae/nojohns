import { formatEther } from "viem";
import { useStats } from "../../hooks/useStats";

export function StatsBar() {
  const { totalMatches, totalWagered, uniqueAgents, isLoading, isError } = useStats();

  if (isLoading) {
    return (
      <div className="grid grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-surface-700 rounded-lg p-4 animate-pulse h-20" />
        ))}
      </div>
    );
  }

  const stats = [
    { label: "Matches Played", value: isError ? "\u2014" : totalMatches.toString() },
    { label: "MON Wagered", value: isError ? "\u2014" : `${Number(formatEther(totalWagered)).toFixed(2)}` },
    { label: "Active Agents", value: isError ? "\u2014" : uniqueAgents.toString() },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {stats.map(({ label, value }) => (
        <div key={label} className="bg-surface-700 rounded-lg p-4 text-center">
          <p className="text-2xl font-mono font-bold text-white">{value}</p>
          <p className="text-xs text-gray-400 mt-1 uppercase tracking-wider">{label}</p>
        </div>
      ))}
    </div>
  );
}
