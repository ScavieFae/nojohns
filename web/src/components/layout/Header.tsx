import { Link, useLocation } from "react-router-dom";

const NAV_ITEMS = [
  { path: "/", label: "Home" },
  { path: "/leaderboard", label: "Leaderboard" },
  { path: "/matches", label: "Matches" },
  { path: "/compete", label: "Compete" },
];

export function Header() {
  const { pathname } = useLocation();

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
                pathname === path
                  ? "bg-surface-600 text-white"
                  : "text-gray-400 hover:text-white hover:bg-surface-700"
              }`}
            >
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
