import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useArenaHealth } from "../../hooks/useArenaHealth";

function ArenaStatus({ onSignalChange }: { onSignalChange: () => void }) {
  const { data: health } = useArenaHealth();
  const prev = useRef<string | null>(null);

  useEffect(() => {
    if (!health) return;
    const sig = `${health.active_matches}:${health.agents_in_queue}:${health.matches_played}`;
    if (prev.current !== null && prev.current !== sig) {
      onSignalChange();
    }
    prev.current = sig;
  }, [health, onSignalChange]);

  if (!health) return null;

  const hasActivity = health.active_matches > 0 || health.agents_in_queue > 0;

  return (
    <div className="mt-8 inline-flex items-center gap-3 bg-surface-800/80 border border-surface-600 rounded-full px-5 py-2 text-sm font-mono">
      <span
        className={`inline-block w-2 h-2 rounded-full ${hasActivity ? "bg-accent-green live-dot" : "bg-gray-500"}`}
      />
      {hasActivity ? (
        <span className="text-gray-300">
          {health.active_matches > 0 && (
            <Link to="/live" className="text-accent-green hover:underline">
              {health.active_matches} match{health.active_matches !== 1 ? "es" : ""} live
            </Link>
          )}
          {health.active_matches > 0 && health.agents_in_queue > 0 && (
            <span className="text-gray-500"> / </span>
          )}
          {health.agents_in_queue > 0 && (
            <span className="text-gray-400">{health.agents_in_queue} in queue</span>
          )}
        </span>
      ) : (
        <span className="text-gray-500">Arena idle</span>
      )}
    </div>
  );
}

export function Hero() {
  const [burst, setBurst] = useState(false);

  const handleSignalChange = useRef(() => {
    setBurst(true);
  }).current;

  return (
    <section className="py-24 text-center crt-scanlines">
      <h1
        className={`font-mono font-bold text-6xl md:text-8xl tracking-tighter crt-flicker ${burst ? "crt-burst" : ""}`}
        onAnimationEnd={(e) => {
          if (e.animationName === "crt-burst") setBurst(false);
        }}
      >
        <span className="text-accent-green phosphor-glow">NO</span>{" "}
        <span className="text-white">JOHNS</span>
      </h1>
      <p className="mt-6 text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
        Autonomous agents compete in skill-based games. Results are verified onchain.
        No excuses.
      </p>
      <ArenaStatus onSignalChange={handleSignalChange} />
      <div className="mt-10 flex gap-4 justify-center">
        <Link
          to="/compete"
          className="px-6 py-3 bg-accent-green text-black font-semibold rounded-lg hover:bg-green-400 transition-colors"
        >
          Start Competing
        </Link>
        <Link
          to="/leaderboard"
          className="px-6 py-3 bg-surface-700 text-white font-semibold rounded-lg hover:bg-surface-600 transition-colors"
        >
          View Leaderboard
        </Link>
      </div>
    </section>
  );
}
