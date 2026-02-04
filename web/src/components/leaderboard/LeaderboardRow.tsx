import { AddressDisplay } from "../shared/AddressDisplay";
import type { AgentStats } from "../../types";

interface LeaderboardRowProps {
  agent: AgentStats;
  rank: number;
}

export function LeaderboardRow({ agent, rank }: LeaderboardRowProps) {
  const rankColors: Record<number, string> = {
    1: "text-accent-yellow",
    2: "text-gray-300",
    3: "text-amber-600",
  };

  return (
    <tr className="border-b border-surface-700 hover:bg-surface-700/50 transition-colors">
      <td className="py-3 px-4">
        <span className={`font-mono font-bold ${rankColors[rank] ?? "text-gray-500"}`}>
          #{rank}
        </span>
      </td>
      <td className="py-3 px-4">
        <AddressDisplay address={agent.address} />
      </td>
      <td className="py-3 px-4 font-mono text-sm">
        <span className="text-accent-green">{agent.wins}W</span>
        <span className="text-gray-500"> - </span>
        <span className="text-accent-red">{agent.losses}L</span>
      </td>
      <td className="py-3 px-4 font-mono text-sm text-gray-400">{agent.totalMatches}</td>
      <td className="py-3 px-4 font-mono text-sm">
        <span
          className={
            agent.winRate >= 0.6
              ? "text-accent-green"
              : agent.winRate >= 0.4
                ? "text-gray-400"
                : "text-accent-red"
          }
        >
          {(agent.winRate * 100).toFixed(0)}%
        </span>
      </td>
    </tr>
  );
}
