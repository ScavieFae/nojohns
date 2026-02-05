/**
 * Live match viewer page
 *
 * Shows live matches when WebSocket endpoint is available,
 * or allows loading replay files for testing.
 */

import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { MeleeViewer } from "../components/viewer/MeleeViewer";
import { useLiveMatch } from "../hooks/useLiveMatch";
import { useArenaHealth } from "../hooks/useArenaHealth";
import { loadReplayFile, type ParsedReplay } from "../lib/replayParser";
import { useReplayPlayback } from "../hooks/useReplayPlayback";

function LiveMatchViewer({ matchId }: { matchId: string }) {
  const { status, matchInfo, currentFrame, error, gameScore } = useLiveMatch(matchId);

  if (status === "connecting") {
    return (
      <div className="flex items-center justify-center h-96 bg-surface-800 rounded-lg">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-400">Connecting to match {matchId}...</p>
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

function ReplayViewer() {
  const [replay, setReplay] = useState<ParsedReplay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playbackState, controls] = useReplayPlayback(replay);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setError(null);
      const parsed = await loadReplayFile(file);
      setReplay(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load replay");
      setReplay(null);
    }
  };

  if (!replay) {
    return (
      <div className="flex flex-col items-center justify-center h-96 bg-surface-800 rounded-lg border-2 border-dashed border-surface-600">
        <p className="text-gray-400 mb-4">Load a .slp replay file to view</p>
        <label className="px-4 py-2 bg-surface-700 text-white rounded cursor-pointer hover:bg-surface-600 transition-colors">
          Choose File
          <input
            type="file"
            accept=".slp"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>
        {error && <p className="text-red-400 mt-4 text-sm">{error}</p>}
      </div>
    );
  }

  return (
    <div>
      {/* Replay info */}
      <div className="flex justify-between items-center mb-4 px-4">
        <div className="flex items-center gap-4">
          {replay.settings.players.map((p, i) => (
            <span key={p.port} className="font-mono text-gray-300">
              {i > 0 && <span className="text-gray-600 mx-2">vs</span>}
              {p.displayName}
            </span>
          ))}
        </div>
        <div className="text-gray-500 text-sm">
          Frame {playbackState.currentFrameIndex + 1} / {playbackState.totalFrames}
        </div>
      </div>

      {/* Viewer */}
      <MeleeViewer frame={playbackState.currentFrame} />

      {/* Playback controls */}
      <div className="mt-4 flex items-center justify-center gap-4">
        <button
          onClick={controls.stepBackward}
          className="px-3 py-1 bg-surface-700 rounded hover:bg-surface-600"
        >
          ←
        </button>
        <button
          onClick={controls.togglePlayPause}
          className="px-4 py-2 bg-accent-green text-black rounded font-semibold hover:bg-green-400"
        >
          {playbackState.isPlaying ? "Pause" : "Play"}
        </button>
        <button
          onClick={controls.stepForward}
          className="px-3 py-1 bg-surface-700 rounded hover:bg-surface-600"
        >
          →
        </button>
        <select
          value={playbackState.speed}
          onChange={(e) => controls.setSpeed(parseFloat(e.target.value))}
          className="px-2 py-1 bg-surface-700 rounded text-sm"
        >
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1">1x</option>
          <option value="2">2x</option>
        </select>
      </div>

      {/* Seek bar */}
      <div className="mt-4 px-4">
        <input
          type="range"
          min={0}
          max={playbackState.totalFrames - 1}
          value={playbackState.currentFrameIndex}
          onChange={(e) => controls.seek(parseInt(e.target.value, 10))}
          className="w-full accent-accent-green"
        />
      </div>

      {/* Load different replay */}
      <div className="mt-4 text-center">
        <label className="text-gray-500 text-sm cursor-pointer hover:text-gray-400">
          Load different replay
          <input
            type="file"
            accept=".slp"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>
      </div>
    </div>
  );
}

export function LivePage() {
  const { matchId } = useParams<{ matchId?: string }>();
  const { data: health } = useArenaHealth();

  const activeMatches = health?.active_matches ?? 0;

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">
          {matchId ? "Live Match" : "Match Viewer"}
        </h1>
        <Link to="/" className="text-gray-400 hover:text-white">
          ← Back
        </Link>
      </div>

      {matchId ? (
        // Viewing a specific live match
        <LiveMatchViewer matchId={matchId} />
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
                <span className="text-gray-400">No live matches right now</span>
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

          {/* Replay viewer */}
          <div className="mb-4">
            <h2 className="text-xl font-semibold mb-2">Replay Viewer</h2>
            <p className="text-gray-500 text-sm mb-4">
              Load a .slp replay file to test the viewer, or click a live match
              above to spectate in real-time.
            </p>
          </div>
          <ReplayViewer />
        </div>
      )}
    </div>
  );
}
