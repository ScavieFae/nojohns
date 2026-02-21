/**
 * Live match viewer page
 *
 * Shows live matches when WebSocket endpoint is available,
 * or allows loading replay files for testing.
 */

import { useParams, Link } from "react-router-dom";
import { MeleeViewer } from "../components/viewer/MeleeViewer";
import { PredictionWidget } from "../components/prediction/PredictionWidget";
import { useLiveMatch } from "../hooks/useLiveMatch";
import { useArenaHealth } from "../hooks/useArenaHealth";
import { useMatchEvents } from "../hooks/useMatchEvents";
import { AddressDisplay } from "../components/shared/AddressDisplay";
import { LiveBetFeed } from "../components/prediction/LiveBetFeed";
import { explorerLink } from "../lib/addresses";

function LiveMatchViewer({ matchId }: { matchId: string }) {
  const { status, matchInfo, currentFrame, error, gameScore } = useLiveMatch(matchId);

  if (status === "connecting" || status === "disconnected") {
    return (
      <div className="flex items-center justify-center h-96 bg-surface-800 rounded-lg">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-400">Connecting to match...</p>
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex items-center justify-center h-96 bg-surface-800 rounded-lg">
        <div className="text-center">
          <p className="text-red-400 mb-2">Connection failed</p>
          <p className="text-gray-500 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (status === "ended") {
    // Check if we actually watched the match (have matchInfo) or connected after it ended
    const watchedMatch = matchInfo !== null;
    return (
      <div className="flex items-center justify-center h-96 bg-surface-800 rounded-lg">
        <div className="text-center">
          <p className="text-accent-green text-xl mb-2">Match Complete</p>
          {watchedMatch ? (
            <p className="text-gray-400">
              Final Score: {gameScore[0]} - {gameScore[1]}
            </p>
          ) : (
            <p className="text-gray-500 text-sm">
              {error || "This match has already ended"}
            </p>
          )}
          <Link
            to="/live"
            className="mt-4 inline-block text-accent-green hover:underline text-sm"
          >
            ← Back to live matches
          </Link>
        </div>
      </div>
    );
  }

  if (!currentFrame || !matchInfo) {
    return (
      <div className="flex items-center justify-center h-96 bg-surface-800 rounded-lg">
        <p className="text-gray-400">Waiting for match data...</p>
      </div>
    );
  }

  return (
    <div>
      {/* Match info header */}
      <div className="flex justify-between items-center mb-4 px-4">
        <div className="flex items-center gap-4">
          {matchInfo.players.map((p, i) => (
            <span key={p.port} className="font-mono text-gray-300">
              {i > 0 && <span className="text-gray-600 mx-2">vs</span>}
              {p.displayName || p.connectCode}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-red-400 text-sm font-mono">LIVE</span>
          <span className="text-gray-500 text-sm ml-2">
            Game Score: {gameScore[0]} - {gameScore[1]}
          </span>
        </div>
      </div>

      {/* Viewer */}
      <MeleeViewer frame={currentFrame} />
    </div>
  );
}


export function LivePage() {
  const { matchId } = useParams<{ matchId?: string }>();
  const { data: health } = useArenaHealth();

  const activeMatches = health?.active_matches ?? 0;

  return (
    <div className={`mx-auto px-4 py-8 ${matchId ? "max-w-6xl" : "max-w-4xl"}`}>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">
          {matchId ? "Live Match" : "Match Viewer"}
        </h1>
        <Link to="/" className="text-gray-400 hover:text-white">
          ← Back
        </Link>
      </div>

      {matchId ? (
        // Two-column: viewer + prediction sidebar
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 min-w-0">
            <LiveMatchViewer matchId={matchId} />
          </div>
          <div className="lg:w-72 flex-shrink-0 space-y-4">
            <PredictionWidget matchId={matchId} />
            <LiveBetFeed />
          </div>
        </div>
      ) : (
        // No match specified - show status and replay viewer
        <div>
          {/* Live status */}
          <div className="mb-8 p-4 bg-surface-800 rounded-lg">
            <div className="flex items-center gap-3">
              <span
                className={`w-3 h-3 rounded-full ${
                  activeMatches > 0 ? "bg-accent-green animate-pulse" : "bg-gray-600"
                }`}
              />
              {activeMatches > 0 ? (
                <span className="text-accent-green">
                  {activeMatches} match{activeMatches !== 1 ? "es" : ""} in progress
                </span>
              ) : (
                <span className="text-gray-400">
                  No live matches right now —{" "}
                  <Link to="/demo" className="text-accent-green hover:underline">
                    watch a demo
                  </Link>
                </span>
              )}
            </div>
            {health?.live_match_ids && health.live_match_ids.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-gray-400 text-sm">Click to spectate:</p>
                {health.live_match_ids.map((id) => (
                  <Link
                    key={id}
                    to={`/live/${id}`}
                    className="block p-3 bg-surface-700 rounded hover:bg-surface-600 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                      <span className="font-mono text-sm">{id}</span>
                      <span className="text-gray-500 text-xs ml-auto">LIVE</span>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>

          {/* Live bet feed */}
          <div className="mb-8">
            <LiveBetFeed />
          </div>

          {/* Recent matches */}
          <RecentMatches />
        </div>
      )}
    </div>
  );
}

function RecentMatches() {
  const { data: matches, isLoading } = useMatchEvents();

  const recent = matches
    ?.sort((a, b) => Number(b.timestamp - a.timestamp))
    .slice(0, 5);

  return (
    <div className="p-4 bg-surface-800 rounded-lg">
      <h2 className="text-sm font-bold text-gray-400 mb-4">Recent Matches</h2>
      {isLoading ? (
        <div className="animate-pulse space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-10 bg-surface-700 rounded" />
          ))}
        </div>
      ) : !recent || recent.length === 0 ? (
        <p className="text-gray-500 text-sm">No matches recorded yet</p>
      ) : (
        <div className="space-y-2">
          {recent.map((m) => {
            const date = new Date(Number(m.timestamp) * 1000);
            const ago = timeAgo(date);
            return (
              <a
                key={m.matchId}
                href={explorerLink("tx", m.transactionHash ?? "")}
                target="_blank"
                rel="noopener noreferrer"
                className="block p-3 bg-surface-700 rounded hover:bg-surface-600 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 min-w-0">
                    <AddressDisplay address={m.winner} link={false} />
                    <span className="text-accent-green text-xs">
                      {m.winnerScore}
                    </span>
                    <span className="text-gray-600">-</span>
                    <span className="text-red-400 text-xs">
                      {m.loserScore}
                    </span>
                    <AddressDisplay address={m.loser} link={false} />
                  </div>
                  <span className="text-gray-500 text-xs flex-shrink-0 ml-2">
                    {ago}
                  </span>
                </div>
              </a>
            );
          })}
          <Link
            to="/matches"
            className="block text-center text-sm text-gray-400 hover:text-accent-green transition-colors pt-2"
          >
            View all matches →
          </Link>
        </div>
      )}
    </div>
  );
}

function timeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
