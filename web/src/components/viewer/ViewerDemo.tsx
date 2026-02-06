/**
 * Demo page for testing MeleeViewer
 * Supports loading .slp replay files for accurate playback testing
 */

import { useState, useCallback, useRef } from "react";
import { MeleeViewer } from "./MeleeViewer";
import { loadReplayFile, type ParsedReplay } from "../../lib/replayParser";
import { useReplayPlayback } from "../../hooks/useReplayPlayback";
import { preloadAnimations } from "../../lib/animationCache";

export function ViewerDemo() {
  const [replay, setReplay] = useState<ParsedReplay | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [playback, controls] = useReplayPlayback(replay);

  // Handle file selection
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const parsed = await loadReplayFile(file);
      console.log("Loaded replay:", parsed.settings);

      // Preload animations for characters in the replay
      const externalIds = parsed.settings.players.map((p) => p.characterId);
      await preloadAnimations(externalIds);

      setReplay(parsed);
    } catch (err) {
      console.error("Failed to load replay:", err);
      setError(err instanceof Error ? err.message : "Failed to load replay");
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Format time from frame number
  const formatTime = (frameIndex: number) => {
    const seconds = Math.floor(frameIndex / 60);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="p-8 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">Melee Viewer</h2>
        <div className="flex gap-2 items-center">
          <input
            ref={fileInputRef}
            type="file"
            accept=".slp"
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="px-4 py-2 bg-surface-700 text-white rounded font-mono text-sm hover:bg-surface-600 disabled:opacity-50"
          >
            {isLoading ? "Loading..." : "Load .slp"}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-accent-red/10 border border-accent-red/30 rounded-lg px-4 py-3">
          <span className="text-accent-red text-sm">{error}</span>
        </div>
      )}

      {/* Match info */}
      {replay && (
        <div className="flex gap-4 text-sm">
          {replay.settings.players.map((player, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-gray-400">P{player.port}:</span>
              <span className="text-white font-mono">{player.displayName}</span>
              <span className="text-gray-500">
                ({getCharacterName(player.characterId)})
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Viewer */}
      <MeleeViewer frame={playback.currentFrame} />

      {/* Playback controls */}
      {replay && (
        <div className="space-y-2">
          {/* Timeline */}
          <div className="flex items-center gap-4">
            <span className="text-gray-400 font-mono text-sm w-16">
              {formatTime(playback.currentFrameIndex)}
            </span>
            <input
              type="range"
              min={0}
              max={playback.totalFrames - 1}
              value={playback.currentFrameIndex}
              onChange={(e) => controls.seek(parseInt(e.target.value, 10))}
              className="flex-1 h-2 bg-surface-700 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-gray-400 font-mono text-sm w-16 text-right">
              {formatTime(playback.totalFrames)}
            </span>
          </div>

          {/* Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={controls.stepBackward}
              className="px-3 py-2 bg-surface-700 text-white rounded font-mono text-sm hover:bg-surface-600"
              title="Step backward"
            >
              ⏮
            </button>
            <button
              onClick={controls.togglePlayPause}
              className="px-4 py-2 bg-accent-blue text-white rounded font-mono text-sm hover:bg-accent-blue/80 min-w-[80px]"
            >
              {playback.isPlaying ? "Pause" : "Play"}
            </button>
            <button
              onClick={controls.stepForward}
              className="px-3 py-2 bg-surface-700 text-white rounded font-mono text-sm hover:bg-surface-600"
              title="Step forward"
            >
              ⏭
            </button>

            <div className="ml-4 flex items-center gap-2">
              <span className="text-gray-400 text-sm">Speed:</span>
              {[0.25, 0.5, 1, 2].map((s) => (
                <button
                  key={s}
                  onClick={() => controls.setSpeed(s)}
                  className={`px-2 py-1 rounded font-mono text-xs ${
                    playback.speed === s
                      ? "bg-accent-blue text-white"
                      : "bg-surface-700 text-gray-300 hover:bg-surface-600"
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Info */}
      <div className="text-sm text-gray-500 font-mono">
        {replay ? (
          <>
            Frame {playback.currentFrameIndex} / {playback.totalFrames} |{" "}
            Stage: {getStageName(replay.settings.stageId)} |{" "}
            Animations: SlippiLab (MIT)
          </>
        ) : (
          <>Load a .slp replay file to test the viewer</>
        )}
      </div>
    </div>
  );
}

// Helper functions
function getCharacterName(externalId: number): string {
  const names = [
    "Falcon", "DK", "Fox", "G&W", "Kirby", "Bowser", "Link", "Luigi",
    "Mario", "Marth", "Mewtwo", "Ness", "Peach", "Pikachu", "ICs",
    "Puff", "Samus", "Yoshi", "Zelda", "Sheik", "Falco", "YLink",
    "Doc", "Roy", "Pichu", "Ganon",
  ];
  return names[externalId] ?? `Char${externalId}`;
}

function getStageName(stageId: number): string {
  const stages: Record<number, string> = {
    2: "Fountain of Dreams",
    3: "Pokemon Stadium",
    8: "Yoshi's Story",
    28: "Dream Land",
    31: "Battlefield",
    32: "Final Destination",
  };
  return stages[stageId] ?? `Stage ${stageId}`;
}
