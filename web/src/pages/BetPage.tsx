import { useState, useCallback, useEffect, useRef } from "react";
import { parseEther, formatEther } from "viem";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { usePrivyWallet } from "../hooks/usePrivyWallet";
import { useCurrentTournamentMatch } from "../hooks/useCurrentTournamentMatch";
import { usePredictionPool, useUserPosition, PoolStatus } from "../hooks/usePredictionPool";
import { publicClient } from "../viem";
import { CONTRACTS, ARENA_URL } from "../config";
import { predictionPoolAbi } from "../abi/predictionPool";

// Fixed bet amount — one tap, one bet
const BET_AMOUNT = "0.05";
const BET_WEI = parseEther(BET_AMOUNT);

// How many bets a spectator can place with their funded balance
function betsRemaining(balanceWei: bigint): number {
  if (BET_WEI === 0n) return 0;
  // Leave ~0.01 MON for gas headroom
  const usable = balanceWei > parseEther("0.01") ? balanceWei - parseEther("0.01") : 0n;
  return Number(usable / BET_WEI);
}

/** Read MON balance for an address. */
function useBalance(address: `0x${string}` | null) {
  return useQuery({
    queryKey: ["balance", address],
    queryFn: () => publicClient.getBalance({ address: address! }),
    enabled: !!address,
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Character icons — simple emoji fallbacks per character name
// ─────────────────────────────────────────────────────────────────────────────

const CHARACTER_COLORS: Record<string, string> = {
  FOX:        "text-yellow-400",
  FALCO:      "text-blue-400",
  MARTH:      "text-red-400",
  SHEIK:      "text-purple-400",
  JIGGLYPUFF: "text-pink-400",
  PIKACHU:    "text-yellow-300",
  CFALCON:    "text-orange-400",
  SAMUS:      "text-green-400",
  GANONDORF:  "text-violet-400",
  LUIGI:      "text-green-300",
  MARIO:      "text-red-300",
  YLINK:      "text-cyan-400",
};

function charColor(character: string): string {
  return CHARACTER_COLORS[character.toUpperCase()] ?? "text-white";
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function SignInScreen({ onLogin }: { onLogin: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
      <div className="mb-8">
        <div className="text-6xl mb-4">🎮</div>
        <h1 className="text-3xl font-black tracking-tight mb-3">Fight Night</h1>
        <div className="space-y-1.5 text-gray-400">
          <p className="text-base">AI agents are fighting in Melee. Pick a winner.</p>
          <p className="text-base">Sign in to get 2 free bets.</p>
          <p className="text-sm text-gray-600">No crypto knowledge needed.</p>
        </div>
      </div>
      <button
        onClick={onLogin}
        className="w-full max-w-sm py-4 rounded-xl bg-accent-green text-black font-bold text-lg hover:bg-accent-green/90 transition-colors"
      >
        Sign in to bet
      </button>
      <p className="text-gray-600 text-sm mt-4">Email, Google, or your own wallet</p>
    </div>
  );
}

function LoadingScreen({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
      <div className="w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full animate-spin mb-4" />
      <p className="text-gray-400">{message}</p>
    </div>
  );
}

function NoMatchScreen() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
      <div className="text-5xl mb-4">⏳</div>
      <h2 className="text-xl font-bold mb-2">No match in progress</h2>
      <p className="text-gray-400">Betting opens when the next match starts.</p>
      <p className="text-gray-600 text-sm mt-2">This page will update automatically.</p>
    </div>
  );
}

function OutOfFundsScreen() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
      <div className="text-5xl mb-4">💸</div>
      <h2 className="text-xl font-bold mb-2">You're out of bets</h2>
      <p className="text-gray-400">No MON left to place more bets tonight.</p>
      <p className="text-gray-600 text-sm mt-3">You can still cheer!</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Odds bar (no addresses — fighter names + characters)
// ─────────────────────────────────────────────────────────────────────────────

function OddsBar({
  nameA,
  nameB,
  totalA,
  totalB,
}: {
  nameA: string;
  nameB: string;
  totalA: bigint;
  totalB: bigint;
}) {
  const total = totalA + totalB;
  const pctA = total > 0n ? Number((totalA * 100n) / total) : 50;
  const pctB = total > 0n ? 100 - pctA : 50;

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-2 font-semibold">
        <span className="text-accent-green truncate mr-2">{nameA}</span>
        <span className="text-purple-400 truncate ml-2 text-right">{nameB}</span>
      </div>
      <div className="flex h-8 rounded-full overflow-hidden bg-surface-700">
        <div
          className="bg-accent-green/80 flex items-center justify-center text-xs font-bold text-black transition-all duration-700"
          style={{ width: `${Math.max(pctA, 8)}%` }}
        >
          {total > 0n && `${pctA}%`}
        </div>
        <div
          className="bg-purple-500/80 flex items-center justify-center text-xs font-bold text-white transition-all duration-700"
          style={{ width: `${Math.max(pctB, 8)}%` }}
        >
          {total > 0n && `${pctB}%`}
        </div>
      </div>
      {total > 0n && (
        <p className="text-center text-xs text-gray-500 mt-1">
          {formatEther(total)} MON in the pool
        </p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main BetPage
// ─────────────────────────────────────────────────────────────────────────────

export function BetPage() {
  const { ready, isAuthenticated, login, account, getWalletClient, isEmbeddedWallet } = usePrivyWallet();
  const { match, isLoading: matchLoading } = useCurrentTournamentMatch();
  const queryClient = useQueryClient();

  // Pool state
  const { pool, poolId, isLoading: poolLoading } = usePredictionPool(
    match?.arena_match_id ?? undefined
  );
  const position = useUserPosition(poolId, account);
  const { data: balance } = useBalance(account);

  // Faucet: fund new embedded wallets once on sign-in
  const faucetCalled = useRef(false);
  useEffect(() => {
    if (!isAuthenticated || !account || !isEmbeddedWallet || faucetCalled.current) return;
    faucetCalled.current = true;
    fetch(`${ARENA_URL}/faucet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ address: account }),
    }).catch(() => {
      // Faucet failure is non-fatal — user can still browse but may not be able to bet
      faucetCalled.current = false; // allow retry on next render
    });
  }, [isAuthenticated, account, isEmbeddedWallet]);

  // Transaction state
  const [txPending, setTxPending] = useState(false);
  const [txError, setTxError] = useState<string | null>(null);
  const [txSuccess, setTxSuccess] = useState<string | null>(null); // "A" or "B"

  const placeBet = useCallback(
    async (betOnA: boolean) => {
      if (poolId === undefined) return;
      setTxError(null);
      setTxPending(true);
      try {
        const client = await getWalletClient();
        if (!client || !account) throw new Error("No wallet");

        const hash = await client.writeContract({
          address: CONTRACTS.predictionPool as `0x${string}`,
          abi: predictionPoolAbi,
          functionName: "bet",
          args: [poolId, betOnA],
          value: BET_WEI,
          account,
        });

        await publicClient.waitForTransactionReceipt({ hash });
        setTxSuccess(betOnA ? "A" : "B");
        // Refresh pool + position + balance
        queryClient.invalidateQueries({ queryKey: ["predictionPool"] });
        queryClient.invalidateQueries({ queryKey: ["predictionPosition"] });
        queryClient.invalidateQueries({ queryKey: ["balance", account] });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Transaction failed";
        setTxError(msg.length > 120 ? msg.slice(0, 120) + "…" : msg);
      } finally {
        setTxPending(false);
      }
    },
    [poolId, account, getWalletClient, queryClient]
  );

  const claimPayout = useCallback(
    async () => {
      if (poolId === undefined || !pool) return;
      setTxError(null);
      setTxPending(true);
      try {
        const client = await getWalletClient();
        if (!client || !account) throw new Error("No wallet");

        const functionName = pool.status === PoolStatus.Cancelled ? "claimRefund" : "claim";
        const hash = await client.writeContract({
          address: CONTRACTS.predictionPool as `0x${string}`,
          abi: predictionPoolAbi,
          functionName,
          args: [poolId],
          account,
        });

        await publicClient.waitForTransactionReceipt({ hash });
        queryClient.invalidateQueries({ queryKey: ["predictionPosition"] });
        queryClient.invalidateQueries({ queryKey: ["balance", account] });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Claim failed";
        setTxError(msg.length > 120 ? msg.slice(0, 120) + "…" : msg);
      } finally {
        setTxPending(false);
      }
    },
    [poolId, pool, account, getWalletClient, queryClient]
  );

  // ─── Loading states ───────────────────────────────────────────────────────

  if (!ready) {
    return <LoadingScreen message="Loading…" />;
  }

  if (!isAuthenticated) {
    return <SignInScreen onLogin={login} />;
  }

  if (matchLoading) {
    return <LoadingScreen message="Looking for a match…" />;
  }

  if (!match) {
    return <NoMatchScreen />;
  }

  // ─── Derived state ────────────────────────────────────────────────────────

  const remaining = balance !== undefined ? betsRemaining(balance) : null;
  const isOpen = pool?.status === PoolStatus.Open;
  const isResolved = pool?.status === PoolStatus.Resolved;
  const isCancelled = pool?.status === PoolStatus.Cancelled;
  const hasBet = position && (position.betOnA > 0n || position.betOnB > 0n);
  const canClaim = position && position.claimable > 0n;

  const betOnA = position?.betOnA ?? 0n;
  const betOnB = position?.betOnB ?? 0n;
  const myBetLabel =
    betOnA > 0n
      ? `You bet on ${match.entry_a.name}`
      : betOnB > 0n
      ? `You bet on ${match.entry_b.name}`
      : null;

  const winner =
    isResolved && pool
      ? pool.winner.toLowerCase() === match.entry_a.wallet_address?.toLowerCase()
        ? match.entry_a.name
        : match.entry_b.name
      : null;

  const youWon =
    isResolved &&
    ((betOnA > 0n && pool?.winner.toLowerCase() === match.entry_a.wallet_address?.toLowerCase()) ||
      (betOnB > 0n && pool?.winner.toLowerCase() === match.entry_b.wallet_address?.toLowerCase()));

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-surface-900 text-white flex flex-col">
      {/* Header */}
      <div className="border-b border-surface-700 px-4 py-3 flex items-center justify-between">
        <span className="text-xs font-black tracking-widest text-gray-400 uppercase">
          Fight Night
        </span>
        {remaining !== null && (
          <span className="text-xs text-gray-500">
            {remaining} bet{remaining !== 1 ? "s" : ""} remaining
          </span>
        )}
      </div>

      {/* Match */}
      <div className="flex-1 flex flex-col justify-center px-5 py-6 space-y-6 max-w-md mx-auto w-full">
        {/* Fighter names */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 text-left">
            <div className={`text-2xl font-black ${charColor(match.entry_a.character)}`}>
              {match.entry_a.name}
            </div>
            <div className="text-sm text-gray-500 uppercase tracking-wide">
              {match.entry_a.character}
            </div>
          </div>
          <div className="text-gray-600 font-bold text-lg">vs</div>
          <div className="flex-1 text-right">
            <div className={`text-2xl font-black ${charColor(match.entry_b.character)}`}>
              {match.entry_b.name}
            </div>
            <div className="text-sm text-gray-500 uppercase tracking-wide text-right">
              {match.entry_b.character}
            </div>
          </div>
        </div>

        {/* Odds */}
        {pool && (
          <OddsBar
            nameA={match.entry_a.name}
            nameB={match.entry_b.name}
            totalA={pool.totalA}
            totalB={pool.totalB}
          />
        )}
        {poolLoading && !pool && (
          <div className="h-8 bg-surface-700 rounded-full animate-pulse" />
        )}

        {/* Post-bet success message */}
        {txSuccess && (
          <div className="bg-accent-green/10 border border-accent-green/30 rounded-xl p-4 text-center">
            <div className="text-2xl mb-1">🎉</div>
            <p className="text-accent-green font-bold">You're in!</p>
            <p className="text-sm text-gray-400 mt-1">
              Rooting for {txSuccess === "A" ? match.entry_a.name : match.entry_b.name}
            </p>
          </div>
        )}

        {/* Resolution / winner announcement */}
        {isResolved && winner && (
          <div className="rounded-xl p-5 text-center bg-surface-800">
            <div className="text-3xl mb-2">{youWon ? "🏆" : "😢"}</div>
            <p className="font-black text-xl mb-1">
              {winner} wins!
            </p>
            {myBetLabel && (
              <p className="text-gray-400 text-sm">
                {youWon ? "You picked right." : "Better luck next match."}
              </p>
            )}
          </div>
        )}

        {isCancelled && (
          <div className="rounded-xl p-4 text-center bg-surface-800">
            <p className="text-gray-400">This match was cancelled.</p>
          </div>
        )}

        {/* Existing position */}
        {hasBet && isOpen && myBetLabel && !txSuccess && (
          <div className="bg-surface-800 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-400">{myBetLabel}</p>
            <p className="text-xs text-gray-600 mt-1">Waiting on the match…</p>
          </div>
        )}

        {/* Action area */}
        <div className="space-y-3">
          {/* Claim button */}
          {canClaim && (
            <button
              onClick={claimPayout}
              disabled={txPending}
              className="w-full py-4 rounded-xl font-black text-lg bg-accent-green text-black hover:bg-accent-green/90 disabled:opacity-50 transition-all"
            >
              {txPending
                ? "Claiming…"
                : isCancelled
                ? "Claim Refund"
                : `Claim ${formatEther(position!.claimable)} MON`}
            </button>
          )}

          {/* Bet buttons — only if pool is open and user hasn't bet yet */}
          {isOpen && !hasBet && !txSuccess && !canClaim && (
            <>
              {remaining === 0 ? (
                <OutOfFundsScreen />
              ) : (
                <>
                  <p className="text-center text-sm text-gray-500">
                    Pick a side — {BET_AMOUNT} MON
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => placeBet(true)}
                      disabled={txPending}
                      className="py-5 rounded-xl font-black text-base bg-accent-green/20 text-accent-green border border-accent-green/40 hover:bg-accent-green/30 disabled:opacity-40 transition-all active:scale-95"
                    >
                      {txPending ? (
                        <span className="inline-block w-5 h-5 border-2 border-accent-green border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <>
                          <div className="text-lg">{match.entry_a.name}</div>
                          <div className="text-xs opacity-70 mt-0.5">{match.entry_a.character}</div>
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => placeBet(false)}
                      disabled={txPending}
                      className="py-5 rounded-xl font-black text-base bg-purple-500/20 text-purple-400 border border-purple-500/40 hover:bg-purple-500/30 disabled:opacity-40 transition-all active:scale-95"
                    >
                      {txPending ? (
                        <span className="inline-block w-5 h-5 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <>
                          <div className="text-lg">{match.entry_b.name}</div>
                          <div className="text-xs opacity-70 mt-0.5">{match.entry_b.character}</div>
                        </>
                      )}
                    </button>
                  </div>
                </>
              )}
            </>
          )}

          {/* No pool yet */}
          {!pool && !poolLoading && (
            <p className="text-center text-sm text-gray-500">
              Prediction pool opening soon…
            </p>
          )}

          {txError && (
            <p className="text-red-400 text-xs text-center break-words px-2">{txError}</p>
          )}
        </div>

        {/* Round indicator */}
        <p className="text-center text-xs text-gray-600">
          Round {match.round + 1} · Match {match.slot + 1}
        </p>
      </div>
    </div>
  );
}
