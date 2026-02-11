import { formatEther } from "viem";
import { OddsBar } from "./OddsBar";
import { BetForm } from "./BetForm";
import {
  usePredictionPool,
  useUserPosition,
  PoolStatus,
} from "../../hooks/usePredictionPool";
import { usePlaceBet } from "../../hooks/usePlaceBet";
import { useClaimPayout } from "../../hooks/useClaimPayout";
import { useWallet } from "../../hooks/useWallet";

interface PredictionWidgetProps {
  matchId: string;
}

export function PredictionWidget({ matchId }: PredictionWidgetProps) {
  const { account } = useWallet();
  const { pool, poolId, isLoading, hasPool } = usePredictionPool(matchId);
  const position = useUserPosition(poolId, account);
  const { placeBet, isPending: betPending, error: betError } = usePlaceBet();
  const {
    claimPayout,
    isPending: claimPending,
    error: claimError,
  } = useClaimPayout();

  if (isLoading) {
    return (
      <div className="bg-surface-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-400 mb-2">Predictions</h3>
        <div className="animate-pulse h-20 bg-surface-700 rounded" />
      </div>
    );
  }

  if (!hasPool || !pool) {
    return (
      <div className="bg-surface-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-400 mb-2">Predictions</h3>
        <p className="text-xs text-gray-500">No prediction pool for this match</p>
      </div>
    );
  }

  const handleBet = async (betOnA: boolean, amount: string) => {
    if (poolId === undefined) return;
    try {
      await placeBet(poolId, betOnA, amount);
    } catch {
      // Error is shown via betError
    }
  };

  const handleClaim = async () => {
    if (poolId === undefined) return;
    try {
      await claimPayout(poolId, pool.status);
    } catch {
      // Error is shown via claimError
    }
  };

  const isOpen = pool.status === PoolStatus.Open;
  const isResolved = pool.status === PoolStatus.Resolved;
  const isCancelled = pool.status === PoolStatus.Cancelled;

  const hasBet =
    position && (position.betOnA > 0n || position.betOnB > 0n);
  const canClaim = position && position.claimable > 0n;

  return (
    <div className="bg-surface-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-white">Predictions</h3>
        <span
          className={`text-xs px-2 py-0.5 rounded ${
            isOpen
              ? "bg-accent-green/20 text-accent-green"
              : isResolved
              ? "bg-blue-500/20 text-blue-400"
              : "bg-gray-500/20 text-gray-400"
          }`}
        >
          {isOpen ? "Open" : isResolved ? "Resolved" : "Cancelled"}
        </span>
      </div>

      {/* Odds */}
      <OddsBar
        playerA={pool.playerA}
        playerB={pool.playerB}
        totalA={pool.totalA}
        totalB={pool.totalB}
      />

      {/* User position */}
      {account && hasBet && (
        <div className="bg-surface-700 rounded p-3 space-y-1">
          <p className="text-xs text-gray-400">Your position</p>
          {position.betOnA > 0n && (
            <p className="text-xs text-accent-green">
              {formatEther(position.betOnA)} MON on Player A
            </p>
          )}
          {position.betOnB > 0n && (
            <p className="text-xs text-red-400">
              {formatEther(position.betOnB)} MON on Player B
            </p>
          )}
          {isOpen && pool.totalA + pool.totalB > 0n && (
            <p className="text-xs text-gray-500">
              Potential payout:{" "}
              {position.betOnA > 0n && pool.totalA > 0n
                ? formatEther(
                    ((pool.totalA + pool.totalB) * position.betOnA) /
                      pool.totalA
                  )
                : position.betOnB > 0n && pool.totalB > 0n
                ? formatEther(
                    ((pool.totalA + pool.totalB) * position.betOnB) /
                      pool.totalB
                  )
                : "0"}{" "}
              MON
            </p>
          )}
        </div>
      )}

      {/* Actions */}
      {!account ? (
        <div className="text-center space-y-2">
          <p className="text-xs text-gray-400">
            Betting is agent-only for now
          </p>
          <a
            href="https://github.com/ScavieFae/nojohns#quick-start"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block text-xs text-accent-green hover:underline"
          >
            Set up an agent to bet →
          </a>
        </div>
      ) : isOpen ? (
        <BetForm
          playerA={pool.playerA}
          playerB={pool.playerB}
          onBet={handleBet}
          isPending={betPending}
          error={betError}
        />
      ) : canClaim ? (
        <div className="space-y-2">
          <p className="text-xs text-accent-green">
            Claimable: {formatEther(position!.claimable)} MON
          </p>
          <button
            onClick={handleClaim}
            disabled={claimPending}
            className="w-full px-4 py-2 rounded text-sm font-bold bg-accent-green/20 text-accent-green border border-accent-green/30 hover:bg-accent-green/30 disabled:opacity-40 transition-colors"
          >
            {claimPending
              ? "Claiming..."
              : isCancelled
              ? "Claim Refund"
              : "Claim Payout"}
          </button>
          {claimError && (
            <p className="text-red-400 text-xs">{claimError}</p>
          )}
        </div>
      ) : (isResolved || isCancelled) && hasBet ? (
        <p className="text-xs text-gray-500">
          {isResolved ? "No payout — wrong side" : "Already claimed"}
        </p>
      ) : null}
    </div>
  );
}
