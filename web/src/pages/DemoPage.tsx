/**
 * Demo page — auto-plays a saved replay and includes replay uploader
 *
 * Judges can see the viewer in action immediately (auto-play),
 * or upload their own .slp file.
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { MeleeViewer } from "../components/viewer/MeleeViewer";
import {
  parseReplay,
  loadReplayFile,
  type ParsedReplay,
} from "../lib/replayParser";
import { useReplayPlayback } from "../hooks/useReplayPlayback";
import { preloadAnimations } from "../lib/animationCache";

export function DemoPage() {
  const [replay, setReplay] = useState<ParsedReplay | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<"sample" | "upload">("sample");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [playback, controls] = useReplayPlayback(replay);

  // Auto-load the bundled sample replay on mount
  useEffect(() => {
    let cancelled = false;

    async function loadSample() {
      try {
        const res = await fetch("/sample.slp");
        if (!res.ok) throw new Error(`Failed to fetch sample replay (${res.status})`);
        const buffer = await res.arrayBuffer();
        const parsed = parseReplay(buffer);

        if (cancelled) return;

        // Preload character animations
        const externalIds = parsed.settings.players.map((p) => p.characterId);
        await preloadAnimations(externalIds);

        if (cancelled) return;

        setReplay(parsed);
        setIsLoading(false);
        setSource("sample");
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to load sample replay:", err);
        setError(err instanceof Error ? err.message : "Failed to load sample replay");
        setIsLoading(false);
      }
    }

    loadSample();
    return () => { cancelled = true; };
  }, []);

  // Auto-play once the sample replay loads
  useEffect(() => {
    if (replay && source === "sample" && !playback.isPlaying) {
      controls.play();
    }
  }, [replay, source]);

  // Handle user uploading their own replay
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const parsed = await loadReplayFile(file);
      const externalIds = parsed.settings.players.map((p) => p.characterId);
      await preloadAnimations(externalIds);
      setReplay(parsed);
      setSource("upload");
    } catch (err) {
      console.error("Failed to load replay:", err);
      setError(err instanceof Error ? err.message : "Failed to load replay");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const formatTime = (frameIndex: number) => {
    const seconds = Math.floor(frameIndex / 60);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold">Demo</h1>
          <span className="px-2 py-0.5 text-xs font-mono font-bold bg-accent-green/20 text-accent-green rounded">
            DEMO
          </span>
        </div>
        <Link to="/live" className="text-gray-400 hover:text-white text-sm">
          ← Live matches
        </Link>
      </div>

      <p className="text-gray-400 mb-6">
        Watch an AI-vs-AI Melee match played by autonomous agents. Every frame is real gameplay
        from a neural net fighter (Phillip) trained on human replays.
      </p>

      {/* Loading state */}
      {isLoading && !replay && (
        <div className="flex items-center justify-center h-96 bg-surface-800 rounded-lg">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 border-accent-green border-t-transparent rounded-full mx-auto mb-4" />
            <p className="text-gray-400">Loading replay...</p>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !replay && (
        <div className="bg-accent-red/10 border border-accent-red/30 rounded-lg px-4 py-3 mb-4">
          <span className="text-accent-red text-sm">{error}</span>
        </div>
      )}

      {/* Match info */}
      {replay && (
        <div className="flex gap-4 text-sm mb-4">
          {replay.settings.players.map((player, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-gray-400">P{player.port}:</span>
              <span className="text-white font-mono">{player.displayName}</span>
              <span className="text-gray-500">
                ({getCharacterName(player.characterId)})
              </span>
            </div>
          ))}
          <span className="text-gray-600 ml-auto text-xs">
            Stage: {getStageName(replay.settings.stageId)}
          </span>
        </div>
      )}

      {/* Viewer */}
      {replay && <MeleeViewer frame={playback.currentFrame} width={730} height={500} />}

      {/* Playback controls */}
      {replay && (
        <div className="space-y-2 mt-4">
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

          {/* Frame info */}
          <div className="text-sm text-gray-500 font-mono">
            Frame {playback.currentFrameIndex} / {playback.totalFrames} | Animations: SlippiLab (MIT)
          </div>
        </div>
      )}

      {/* Upload section */}
      <div className="mt-8 p-4 bg-surface-800 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-gray-400 mb-1">Upload Your Own Replay</h2>
            <p className="text-gray-500 text-xs">
              Drop any Slippi .slp file to view it in the browser
            </p>
          </div>
          <div>
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
      </div>
    </div>
  );
}

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
