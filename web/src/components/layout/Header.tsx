import { Link, useLocation } from "react-router-dom";
import { useArenaHealth } from "../../hooks/useArenaHealth";

const NAV_ITEMS = [
  { path: "/", label: "Home" },
  { path: "/live", label: "Live" },
  { path: "/leaderboard", label: "Leaderboard" },
  { path: "/matches", label: "Matches" },
  { path: "/compete", label: "Compete" },
];

export function Header() {
  const { pathname } = useLocation();
  const { data: health } = useArenaHealth();
  const hasLiveMatches = (health?.live_match_ids?.length ?? 0) > 0;

  return (
    <header className="border-b border-surface-600 bg-surface-800/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="font-mono font-bold text-xl tracking-tight">
          <span className="text-accent-green">NO</span>{" "}
          <span className="text-white">JOHNS</span>
        </Link>

        <nav className="flex gap-1">
          {NAV_ITEMS.map(({ path, label }) => (
            <Link
              key={path}
              to={path}
              className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                pathname === path || (path === "/live" && pathname.startsWith("/live"))
                  ? "bg-surface-600 text-white"
                  : "text-gray-400 hover:text-white hover:bg-surface-700"
              }`}
            >
              {label}
              {path === "/live" && hasLiveMatches && (
                <span className="ml-1.5 inline-block w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              )}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
