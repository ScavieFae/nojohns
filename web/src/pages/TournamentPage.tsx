import { useState, useEffect, useRef } from "react";
import { usePrivyWallet } from "../hooks/usePrivyWallet";
import {
  useFeaturedTournament,
  useMyEntry,
  usePlayingMatch,
  type TournamentEntry,
  type TournamentData,
} from "../hooks/useTournament";
import { ARENA_URL } from "../config";
import { useQueryClient } from "@tanstack/react-query";

// ─────────────────────────────────────────────────────────────────────────────
// Character data
// ─────────────────────────────────────────────────────────────────────────────

const CHARACTER_COLORS: Record<string, { text: string; glow: string; bg: string }> = {
  FOX:            { text: "text-yellow-400",  glow: "shadow-yellow-400/30",  bg: "bg-yellow-400/10" },
  FALCO:          { text: "text-blue-400",    glow: "shadow-blue-400/30",    bg: "bg-blue-400/10" },
  MARTH:          { text: "text-red-400",     glow: "shadow-red-400/30",     bg: "bg-red-400/10" },
  SHEIK:          { text: "text-purple-400",  glow: "shadow-purple-400/30",  bg: "bg-purple-400/10" },
  JIGGLYPUFF:     { text: "text-pink-400",    glow: "shadow-pink-400/30",    bg: "bg-pink-400/10" },
  PIKACHU:        { text: "text-yellow-300",  glow: "shadow-yellow-300/30",  bg: "bg-yellow-300/10" },
  CAPTAINFALCON:  { text: "text-orange-400",  glow: "shadow-orange-400/30",  bg: "bg-orange-400/10" },
  CFALCON:        { text: "text-orange-400",  glow: "shadow-orange-400/30",  bg: "bg-orange-400/10" },
  SAMUS:          { text: "text-green-400",   glow: "shadow-green-400/30",   bg: "bg-green-400/10" },
  GANONDORF:      { text: "text-violet-400",  glow: "shadow-violet-400/30",  bg: "bg-violet-400/10" },
  LUIGI:          { text: "text-green-300",   glow: "shadow-green-300/30",   bg: "bg-green-300/10" },
  MARIO:          { text: "text-red-300",     glow: "shadow-red-300/30",     bg: "bg-red-300/10" },
  YOSHI:          { text: "text-emerald-400", glow: "shadow-emerald-400/30", bg: "bg-emerald-400/10" },
  PEACH:          { text: "text-pink-300",    glow: "shadow-pink-300/30",    bg: "bg-pink-300/10" },
  DONKEYKONG:     { text: "text-amber-600",   glow: "shadow-amber-600/30",   bg: "bg-amber-600/10" },
  MEWTWO:         { text: "text-purple-300",  glow: "shadow-purple-300/30",  bg: "bg-purple-300/10" },
  ROY:            { text: "text-red-500",     glow: "shadow-red-500/30",     bg: "bg-red-500/10" },
  LINK:           { text: "text-green-500",   glow: "shadow-green-500/30",   bg: "bg-green-500/10" },
  YOUNGLINK:      { text: "text-cyan-400",    glow: "shadow-cyan-400/30",    bg: "bg-cyan-400/10" },
  YLINK:          { text: "text-cyan-400",    glow: "shadow-cyan-400/30",    bg: "bg-cyan-400/10" },
  KIRBY:          { text: "text-pink-400",    glow: "shadow-pink-400/30",    bg: "bg-pink-400/10" },
  BOWSER:         { text: "text-orange-600",  glow: "shadow-orange-600/30",  bg: "bg-orange-600/10" },
  ZELDA:          { text: "text-purple-300",  glow: "shadow-purple-300/30",  bg: "bg-purple-300/10" },
  NESS:           { text: "text-red-400",     glow: "shadow-red-400/30",     bg: "bg-red-400/10" },
  POPO:           { text: "text-blue-300",    glow: "shadow-blue-300/30",    bg: "bg-blue-300/10" },
  DR_MARIO:       { text: "text-white",       glow: "shadow-white/20",       bg: "bg-white/10" },
  MRGAMEANDWATCH: { text: "text-gray-400",    glow: "shadow-gray-400/30",    bg: "bg-gray-400/10" },
  PICHU:          { text: "text-yellow-200",  glow: "shadow-yellow-200/30",  bg: "bg-yellow-200/10" },
};

function charStyle(character: string) {
  return CHARACTER_COLORS[character.toUpperCase()] ?? { text: "text-white", glow: "shadow-white/20", bg: "bg-white/10" };
}

const CHAR_DISPLAY: Record<string, string> = {
  FOX: "Fox", FALCO: "Falco", MARTH: "Marth", SHEIK: "Sheik",
  JIGGLYPUFF: "Puff", PEACH: "Peach", POPO: "Ice Climbers",
  CAPTAINFALCON: "Falcon", CFALCON: "Falcon", PIKACHU: "Pikachu",
  SAMUS: "Samus", DR_MARIO: "Dr. Mario", LINK: "Link",
  YOUNGLINK: "Young Link", YLINK: "Young Link", MEWTWO: "Mewtwo",
  ROY: "Roy", GANONDORF: "Ganondorf", LUIGI: "Luigi", MARIO: "Mario",
  YOSHI: "Yoshi", DONKEYKONG: "DK", NESS: "Ness", PICHU: "Pichu",
  MRGAMEANDWATCH: "G&W", KIRBY: "Kirby", BOWSER: "Bowser", ZELDA: "Zelda",
};

function charName(c: string) {
  return CHAR_DISPLAY[c.toUpperCase()] ?? c;
}

const CHARACTERS = [
  "FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "PEACH",
  "CAPTAINFALCON", "PIKACHU", "SAMUS", "GANONDORF", "LUIGI",
  "MARIO", "YOSHI", "DONKEYKONG", "LINK", "YOUNGLINK",
  "MEWTWO", "ROY", "NESS", "DR_MARIO", "KIRBY", "BOWSER", "ZELDA",
];

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
// Fighter card
// ─────────────────────────────────────────────────────────────────────────────

function FighterCard({ entry, tournament }: { entry: TournamentEntry; tournament: TournamentData }) {
  const style = charStyle(entry.character);

  // Calculate record from bracket
  let wins = 0;
  let losses = 0;
  if (tournament.bracket?.rounds) {
    for (const round of tournament.bracket.rounds) {
      for (const match of round) {
        if (match.status !== "complete" && match.status !== "coinflip") continue;
        const isA = match.entry_a?.name === entry.name;
        const isB = match.entry_b?.name === entry.name;
        if (!isA && !isB) continue;
        if (match.winner?.name === entry.name) {
          wins++;
        } else if (match.winner) {
          losses++;
        }
      }
    }
  }

  const isEliminated = losses > 0;
  const rounds = tournament.bracket?.rounds ?? [];
  const finalRound = rounds.length > 0 ? rounds[rounds.length - 1] : [];
  const isChampion = tournament.status === "complete" &&
    finalRound.length === 1 && finalRound[0]?.winner?.name === entry.name;

  return (
    <div className={`rounded-2xl border border-surface-600 ${style.bg} p-6 relative overflow-hidden`}>
      {/* Scan line overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-30"
        style={{
          background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px)",
        }}
      />

      <div className="relative z-10">
        {isChampion && (
          <div className="text-center mb-3">
            <span className="text-yellow-400 text-xs font-black tracking-widest uppercase"
              style={{ fontFamily: "'Orbitron', sans-serif" }}>
              Champion
            </span>
          </div>
        )}

        <div className="text-center mb-1">
          <span className={`text-xs tracking-widest uppercase ${style.text}`}
            style={{ fontFamily: "'Orbitron', sans-serif" }}>
            {charName(entry.character)}
          </span>
        </div>

        <h2
          className="text-3xl font-black text-center mb-4"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          {entry.name}
        </h2>

        <div className="flex justify-center gap-6 text-center">
          <div>
            <div className="text-2xl font-black text-accent-green">{wins}</div>
            <div className="text-xs text-gray-500 uppercase tracking-wider">Wins</div>
          </div>
          <div className="w-px bg-surface-600" />
          <div>
            <div className="text-2xl font-black text-red-400">{losses}</div>
            <div className="text-xs text-gray-500 uppercase tracking-wider">Losses</div>
          </div>
        </div>

        {isEliminated && !isChampion && (
          <div className="mt-4 text-center">
            <span className="text-red-400/70 text-xs font-bold uppercase tracking-wider">Eliminated</span>
          </div>
        )}
        {!isEliminated && !isChampion && tournament.status === "active" && (
          <div className="mt-4 text-center">
            <span className="text-accent-green text-xs font-bold uppercase tracking-wider animate-pulse">
              Still in it
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration form
// ─────────────────────────────────────────────────────────────────────────────

function RegisterForm({
  tournamentId,
  email,
  walletAddress,
  onRegistered,
}: {
  tournamentId: string;
  email: string;
  walletAddress: string | null;
  onRegistered: () => void;
}) {
  const [name, setName] = useState("");
  const [character, setCharacter] = useState("RANDOM");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!name.trim()) {
      setError("Pick a fighter name");
      return;
    }

    setSubmitting(true);
    setError(null);

    // If RANDOM, pick one
    const finalChar = character === "RANDOM"
      ? CHARACTERS[Math.floor(Math.random() * CHARACTERS.length)]
      : character;

    try {
      const res = await fetch(`${ARENA_URL}/tournaments/${tournamentId}/self-register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          character: finalChar,
          email,
          wallet_address: walletAddress,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Registration failed" }));
        throw new Error(data.detail || "Registration failed");
      }

      onRegistered();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Registration failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-4">
      <h3
        className="text-lg font-black tracking-tight"
        style={{ fontFamily: "'Orbitron', sans-serif" }}
      >
        Register Your Fighter
      </h3>

      <div>
        <label className="block text-xs text-gray-500 uppercase tracking-wider mb-1.5">
          Fighter Name
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Shadow Falcon"
          maxLength={24}
          className="w-full bg-surface-800 border border-surface-600 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:border-accent-green focus:outline-none transition-colors text-base"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-500 uppercase tracking-wider mb-1.5">
          Character
        </label>
        <div className="grid grid-cols-4 gap-1.5 max-h-48 overflow-y-auto">
          <button
            onClick={() => setCharacter("RANDOM")}
            className={`px-2 py-2 rounded-lg text-xs font-bold transition-all ${
              character === "RANDOM"
                ? "bg-accent-green/20 text-accent-green border border-accent-green/40"
                : "bg-surface-800 text-gray-400 border border-surface-600"
            }`}
          >
            Random
          </button>
          {CHARACTERS.map((c) => {
            const s = charStyle(c);
            return (
              <button
                key={c}
                onClick={() => setCharacter(c)}
                className={`px-2 py-2 rounded-lg text-xs font-bold transition-all truncate ${
                  character === c
                    ? `${s.bg} ${s.text} border border-current`
                    : "bg-surface-800 text-gray-400 border border-surface-600"
                }`}
              >
                {charName(c)}
              </button>
            );
          })}
        </div>
      </div>

      {error && (
        <p className="text-red-400 text-sm text-center">{error}</p>
      )}

      <button
        onClick={handleSubmit}
        disabled={submitting || !name.trim()}
        className="w-full py-4 rounded-xl bg-accent-green text-black font-bold text-lg disabled:opacity-40 active:scale-95 transition-all"
      >
        {submitting ? "Registering..." : "Enter Tournament"}
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard — the three-button hub
// ─────────────────────────────────────────────────────────────────────────────

function Dashboard({
  tournament,
  myEntry,
  email,
  walletAddress,
  onRefetch,
}: {
  tournament: TournamentData;
  myEntry: TournamentEntry | null;
  email: string | null;
  walletAddress: string | null;
  onRefetch: () => void;
}) {
  const [showRegister, setShowRegister] = useState(false);
  const [showFighter, setShowFighter] = useState(false);
  const playingMatch = usePlayingMatch(tournament);

  const regOpen = tournament.status === "registration";
  const hasFighter = !!myEntry;

  // Derive fight button state
  let fightLabel: string;
  let fightAction: (() => void) | null;
  let fightDisabled = false;

  if (hasFighter) {
    fightLabel = "My Fighter";
    fightAction = () => setShowFighter(!showFighter);
  } else if (regOpen) {
    fightLabel = "Register";
    fightAction = () => setShowRegister(!showRegister);
  } else {
    fightLabel = "Registration Closed";
    fightAction = null;
    fightDisabled = true;
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <div className="border-b border-surface-600 px-4 py-3 flex items-center justify-between">
        <div>
          <h1
            className="text-lg font-black tracking-tight"
            style={{ fontFamily: "'Orbitron', sans-serif" }}
          >
            {tournament.name}
          </h1>
          <p className="text-xs text-gray-500">
            {tournament.entries.length} fighters
            {tournament.status === "registration" && " · Registration open"}
            {tournament.status === "active" && " · In progress"}
            {tournament.status === "complete" && " · Complete"}
          </p>
        </div>
        <StatusPill status={tournament.status} />
      </div>

      {/* Current match banner */}
      {playingMatch && (
        <div className="bg-surface-800 border-b border-surface-600 px-4 py-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 bg-accent-green rounded-full animate-pulse" />
            <span className="text-xs text-accent-green font-bold uppercase tracking-wider">
              Now Playing
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm">{playingMatch.entry_a?.name}</span>
            <span className="text-gray-600 text-xs">vs</span>
            <span className="font-bold text-sm">{playingMatch.entry_b?.name}</span>
          </div>
        </div>
      )}

      {/* Three buttons */}
      <div className="flex-1 flex flex-col justify-center px-5 py-6 space-y-4 max-w-md mx-auto w-full">
        {/* Fight button */}
        <button
          onClick={fightAction ?? undefined}
          disabled={fightDisabled}
          className={`w-full py-6 rounded-2xl font-black text-xl transition-all active:scale-[0.97] ${
            fightDisabled
              ? "bg-surface-800 text-gray-600 border border-surface-600"
              : hasFighter
              ? "bg-accent-green/15 text-accent-green border border-accent-green/30 hover:bg-accent-green/20"
              : "bg-accent-green/15 text-accent-green border border-accent-green/30 hover:bg-accent-green/20"
          }`}
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          {fightLabel}
        </button>

        {/* Watch button */}
        <a
          href="/tournament"
          target="_blank"
          rel="noopener noreferrer"
          className="block w-full py-6 rounded-2xl font-black text-xl text-center bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 transition-all active:scale-[0.97]"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          Watch
        </a>

        {/* Bet button */}
        <a
          href="/tournament/bet"
          className="block w-full py-6 rounded-2xl font-black text-xl text-center bg-purple-500/15 text-purple-400 border border-purple-500/30 hover:bg-purple-500/20 transition-all active:scale-[0.97]"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          Bet
        </a>
      </div>

      {/* Fighter card (slide down) */}
      {showFighter && myEntry && (
        <div className="px-5 pb-6 max-w-md mx-auto w-full">
          <FighterCard entry={myEntry} tournament={tournament} />
        </div>
      )}

      {/* Registration form (slide down) */}
      {showRegister && email && (
        <div className="px-5 pb-6 max-w-md mx-auto w-full">
          <RegisterForm
            tournamentId={tournament.id}
            email={email}
            walletAddress={walletAddress}
            onRegistered={() => {
              setShowRegister(false);
              onRefetch();
            }}
          />
        </div>
      )}

      {/* Footer with wallet info */}
      <div className="border-t border-surface-600 px-4 py-3 text-center">
        <p className="text-xs text-gray-600">
          {email && <>Signed in as {email}</>}
        </p>
      </div>
    </div>
  );
}

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
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export function TournamentPage() {
  const { ready, isAuthenticated, login, account, user, isEmbeddedWallet } = usePrivyWallet();
  const { tournament, isLoading: tournLoading } = useFeaturedTournament();
  const queryClient = useQueryClient();

  // Extract email from Privy user
  const email = user?.email?.address ?? null;

  // Look up this user's entry
  const { entry: myEntry, refetch: refetchEntry } = useMyEntry(
    tournament?.id,
    email,
    account,
  );

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

  // Loading
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

  // Loading tournament
  if (tournLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
        <div className="w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-gray-400 text-sm">Loading tournament...</p>
      </div>
    );
  }

  // No tournament
  if (!tournament) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen px-6 text-center">
        <h2 className="text-xl font-bold mb-2" style={{ fontFamily: "'Orbitron', sans-serif" }}>
          No Active Tournament
        </h2>
        <p className="text-gray-400 text-sm">Check back when Fight Night starts.</p>
      </div>
    );
  }

  return (
    <Dashboard
      tournament={tournament}
      myEntry={myEntry}
      email={email}
      walletAddress={account}
      onRefetch={() => {
        refetchEntry();
        queryClient.invalidateQueries({ queryKey: ["featuredTournament"] });
      }}
    />
  );
}
