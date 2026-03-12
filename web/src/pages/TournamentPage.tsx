import { useState, useEffect, useRef } from "react";
import { usePrivyWallet } from "../hooks/usePrivyWallet";
import {
  useAllTournaments,
  type TournamentSummary,
} from "../hooks/useTournament";
import { ARENA_URL } from "../config";
import { publicClient } from "../viem";
import { formatEther } from "viem";

// ─────────────────────────────────────────────────────────────────────────────
// Sign-in screen
// ─────────────────────────────────────────────────────────────────────────────

function SignInScreen({ onLogin }: { onLogin: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
      <div className="mb-10">
        <h1
          className="text-4xl font-black tracking-tight mb-1"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          <span className="text-accent-green">FIGHT</span>{" "}
          <span className="text-white">NIGHT</span>
        </h1>
        <p className="text-gray-500 text-sm tracking-widest uppercase" style={{ fontFamily: "'Space Mono', monospace" }}>
          Agentic Smash Bros
        </p>
      </div>

      <button
        onClick={onLogin}
        className="w-full max-w-xs py-4 rounded-xl bg-accent-green text-black font-bold text-lg active:scale-95 transition-all"
      >
        Sign in with Email
      </button>
      <p className="text-gray-600 text-xs mt-3">
        Use the email you signed up with on Luma
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MON balance hook
// ─────────────────────────────────────────────────────────────────────────────

function useMonBalance(address: `0x${string}` | null) {
  const [balance, setBalance] = useState<string | null>(null);

  useEffect(() => {
    if (!address) {
      setBalance(null);
      return;
    }

    let cancelled = false;

    async function fetch() {
      try {
        const raw = await publicClient.getBalance({ address: address! });
        if (!cancelled) {
          const formatted = formatEther(raw);
          // Show up to 2 decimal places, strip trailing zeros
          const num = parseFloat(formatted);
          setBalance(num < 0.01 && num > 0 ? "<0.01" : num.toFixed(2));
        }
      } catch {
        if (!cancelled) setBalance(null);
      }
    }

    fetch();
    const interval = setInterval(fetch, 15_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [address]);

  return balance;
}

// ─────────────────────────────────────────────────────────────────────────────
// Status pill
// ─────────────────────────────────────────────────────────────────────────────

function StatusPill({ status }: { status: string }) {
  const styles: Record<string, string> = {
    registration: "bg-blue-500/10 text-blue-400 border-blue-500/30",
    pending: "bg-gray-500/10 text-gray-400 border-gray-500/30",
    active: "bg-accent-green/10 text-accent-green border-accent-green/30",
    complete: "bg-yellow-500/10 text-yellow-400 border-yellow-500/30",
  };
  const labels: Record<string, string> = {
    registration: "Open",
    pending: "Upcoming",
    active: "Live",
    complete: "Done",
  };

  return (
    <span
      className={`px-3 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${styles[status] ?? styles.pending}`}
      style={{ fontFamily: "'Orbitron', sans-serif" }}
    >
      {labels[status] ?? status}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tournament list
// ─────────────────────────────────────────────────────────────────────────────

function TournamentList({ tournaments }: { tournaments: TournamentSummary[] }) {
  const statusOrder: Record<string, number> = { active: 0, registration: 1, pending: 2, complete: 3 };
  const sorted = [...tournaments].sort(
    (a, b) => (statusOrder[a.status] ?? 9) - (statusOrder[b.status] ?? 9)
  );

  return (
    <div className="space-y-2">
      {sorted.map((t) => (
        <a
          key={t.id}
          href="/tournament"
          className="block w-full text-left bg-surface-800 border border-surface-600 rounded-xl px-4 py-3 hover:border-accent-green/30 transition-colors active:scale-[0.98]"
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm">{t.name}</span>
            <StatusPill status={t.status} />
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {t.entry_count ? `${t.entry_count} fighters` : t.id.slice(0, 8)}
          </p>
        </a>
      ))}
      {tournaments.length === 0 && (
        <p className="text-gray-500 text-sm text-center py-4">No tournaments yet</p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export function TournamentPage() {
  const { ready, isAuthenticated, login, logout, account, user, isEmbeddedWallet } = usePrivyWallet();

  const email = user?.email?.address ?? null;
  const balance = useMonBalance(account);
  const { tournaments, isLoading: listLoading } = useAllTournaments(isAuthenticated);

  // Faucet on first embedded wallet (50 MON for fight night)
  const faucetCalled = useRef(false);
  useEffect(() => {
    if (!isAuthenticated || !account || !isEmbeddedWallet || faucetCalled.current) return;
    faucetCalled.current = true;
    fetch(`${ARENA_URL}/faucet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ address: account }),
    }).catch(() => {
      faucetCalled.current = false;
    });
  }, [isAuthenticated, account, isEmbeddedWallet]);

  // Loading Privy
  if (!ready) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Not signed in
  if (!isAuthenticated) {
    return <SignInScreen onLogin={login} />;
  }

  // Loading tournament list
  if (listLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
        <div className="w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-gray-400 text-sm">Loading...</p>
      </div>
    );
  }

  // Signed in + on Luma → show dashboard
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <div className="border-b border-surface-600 px-4 py-3 flex items-center justify-between">
        <div>
          <h1
            className="text-lg font-black tracking-tight"
            style={{ fontFamily: "'Orbitron', sans-serif" }}
          >
            <span className="text-accent-green">FIGHT</span>{" "}
            <span className="text-white">NIGHT</span>
          </h1>
          <p className="text-xs text-gray-500">{email}</p>
        </div>
        <button onClick={logout} className="text-xs text-gray-500 hover:text-white transition-colors">
          Sign out
        </button>
      </div>

      {/* Balance card */}
      <div className="px-5 pt-5 pb-3 max-w-md mx-auto w-full">
        <div className="bg-surface-800 border border-surface-600 rounded-xl px-5 py-4">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1"
            style={{ fontFamily: "'Space Mono', monospace" }}>
            Your Balance
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className="text-3xl font-black text-white"
              style={{ fontFamily: "'Orbitron', sans-serif" }}
            >
              {balance ?? "—"}
            </span>
            <span className="text-sm text-gray-500">MON</span>
          </div>
          {account && (
            <p className="text-xs text-gray-600 mt-2 font-mono truncate">
              {account}
            </p>
          )}
        </div>
      </div>

      {/* Quick actions */}
      <div className="px-5 pb-3 max-w-md mx-auto w-full grid grid-cols-2 gap-3">
        <a
          href="/tournament"
          className="block py-4 rounded-xl font-bold text-center bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 transition-all active:scale-[0.97]"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          Watch
        </a>
        <a
          href="/tournament/bet"
          className="block py-4 rounded-xl font-bold text-center bg-purple-500/15 text-purple-400 border border-purple-500/30 hover:bg-purple-500/20 transition-all active:scale-[0.97]"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          Bet
        </a>
      </div>

      {/* Tournament list */}
      <div className="px-5 pt-3 pb-6 max-w-md mx-auto w-full">
        <h2
          className="text-sm font-black uppercase tracking-wider text-gray-400 mb-3"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          Tournaments
        </h2>
        <TournamentList tournaments={tournaments} />
      </div>
    </div>
  );
}
