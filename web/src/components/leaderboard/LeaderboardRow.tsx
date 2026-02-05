import { AddressDisplay } from "../shared/AddressDisplay";
import type { AgentStats } from "../../types";

interface LeaderboardRowProps {
  agent: AgentStats;
  rank: number;
}

const rankColors: Record<number, string> = {
  1: "text-accent-yellow",
  2: "text-gray-300",
  3: "text-amber-600",
};

function RankBadge({ rank }: { rank: number }) {
  return (
    <span className={`font-mono font-bold ${rankColors[rank] ?? "text-gray-500"}`}>
      #{rank}
    </span>
  );
}

function WinRateText({ winRate }: { winRate: number }) {
  return (
    <span
      className={
        winRate >= 0.6
          ? "text-accent-green"
          : winRate >= 0.4
            ? "text-gray-400"
            : "text-accent-red"
      }
    >
      {(winRate * 100).toFixed(0)}%
    </span>
  );
}

function EloDisplay({ elo }: { elo: number }) {
  const color =
    elo >= 1600
      ? "text-accent-yellow"
      : elo >= 1500
        ? "text-white"
        : "text-gray-400";
  return <span className={`font-mono font-bold ${color}`}>{elo}</span>;
}

export function LeaderboardRow({ agent, rank }: LeaderboardRowProps) {
  return (
    <tr className="border-b border-surface-700 hover:bg-surface-700/50 transition-colors">
      <td className="py-3 px-4">
        <RankBadge rank={rank} />
      </td>
      <td className="py-3 px-4">
        <AddressDisplay address={agent.address} />
      </td>
      <td className="py-3 px-4">
        <EloDisplay elo={agent.elo} />
      </td>
      <td className="py-3 px-4 font-mono text-sm">
        <span className="text-accent-green">{agent.wins}W</span>
        <span className="text-gray-500"> - </span>
        <span className="text-accent-red">{agent.losses}L</span>
      </td>
      <td className="py-3 px-4 font-mono text-sm text-gray-400">{agent.totalMatches}</td>
      <td className="py-3 px-4 font-mono text-sm">
        <WinRateText winRate={agent.winRate} />
      </td>
    </tr>
  );
}

export function LeaderboardCard({ agent, rank }: LeaderboardRowProps) {
  return (
    <div className="border-b border-surface-700 p-4 space-y-2">
      <div className="flex items-center justify-between">
        <RankBadge rank={rank} />
        <EloDisplay elo={agent.elo} />
      </div>
      <div>
        <AddressDisplay address={agent.address} />
      </div>
      <div className="flex gap-4 text-sm font-mono">
        <span>
          <span className="text-accent-green">{agent.wins}W</span>
          <span className="text-gray-500"> - </span>
          <span className="text-accent-red">{agent.losses}L</span>
        </span>
        <span className="text-gray-500">{agent.totalMatches} games</span>
        <span>
          <WinRateText winRate={agent.winRate} />
        </span>
      </div>
    </div>
  );
}
