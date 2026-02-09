import { formatEther } from "viem";
import { truncateAddress } from "../../lib/addresses";

interface OddsBarProps {
  playerA: `0x${string}`;
  playerB: `0x${string}`;
  totalA: bigint;
  totalB: bigint;
}

export function OddsBar({ playerA, playerB, totalA, totalB }: OddsBarProps) {
  const total = totalA + totalB;
  const pctA = total > 0n ? Number((totalA * 100n) / total) : 50;
  const pctB = total > 0n ? 100 - pctA : 50;
  const totalMon = total > 0n ? formatEther(total) : "0";

  return (
    <div>
      {/* Player labels */}
      <div className="flex justify-between text-xs mb-1">
        <span className="text-accent-green font-mono">
          {truncateAddress(playerA)}
        </span>
        <span className="text-red-400 font-mono">
          {truncateAddress(playerB)}
        </span>
      </div>

      {/* Odds bar */}
      <div className="flex h-6 rounded overflow-hidden bg-surface-700">
        <div
          className="bg-accent-green/70 flex items-center justify-center text-xs font-bold transition-all duration-500"
          style={{ width: `${Math.max(pctA, 5)}%` }}
        >
          {total > 0n && <span>{pctA}%</span>}
        </div>
        <div
          className="bg-red-500/70 flex items-center justify-center text-xs font-bold transition-all duration-500"
          style={{ width: `${Math.max(pctB, 5)}%` }}
        >
          {total > 0n && <span>{pctB}%</span>}
        </div>
      </div>

      {/* Total pool */}
      <div className="text-center text-xs text-gray-500 mt-1">
        Pool: {totalMon} MON
      </div>
    </div>
  );
}
