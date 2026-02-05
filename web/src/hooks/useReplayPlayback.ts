/**
 * Hook for controlling replay playback
 */

import { useState, useCallback, useRef, useEffect } from "react";
import type { ParsedReplay } from "../lib/replayParser";
import type { MatchFrame } from "../components/viewer/MeleeViewer";

export interface PlaybackState {
  isPlaying: boolean;
  currentFrameIndex: number;
  totalFrames: number;
  speed: number; // 1 = normal, 0.5 = half speed, 2 = double
  currentFrame: MatchFrame | null;
}

export interface PlaybackControls {
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  seek: (frameIndex: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  setSpeed: (speed: number) => void;
}

/**
 * Control replay playback with play/pause, seeking, and speed control
 */
export function useReplayPlayback(
  replay: ParsedReplay | null
): [PlaybackState, PlaybackControls] {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [speed, setSpeed] = useState(1);

  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const accumulatorRef = useRef<number>(0);

  // Frame duration in ms (60fps base)
  const frameDuration = (1000 / 60) / speed;

  // Reset when replay changes
  useEffect(() => {
    setCurrentFrameIndex(0);
    setIsPlaying(false);
    accumulatorRef.current = 0;
  }, [replay]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !replay) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }

    const animate = (timestamp: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp;
      }

      const deltaTime = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;
      accumulatorRef.current += deltaTime;

      // Advance frames based on accumulated time
      while (accumulatorRef.current >= frameDuration) {
        accumulatorRef.current -= frameDuration;
        setCurrentFrameIndex((prev) => {
          const next = prev + 1;
          if (next >= replay.totalFrames) {
            setIsPlaying(false);
            return replay.totalFrames - 1;
          }
          return next;
        });
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [isPlaying, replay, frameDuration]);

  // Controls
  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const togglePlayPause = useCallback(() => setIsPlaying((p) => !p), []);

  const seek = useCallback(
    (frameIndex: number) => {
      if (!replay) return;
      const clamped = Math.max(0, Math.min(frameIndex, replay.totalFrames - 1));
      setCurrentFrameIndex(clamped);
      accumulatorRef.current = 0;
    },
    [replay]
  );

  const stepForward = useCallback(() => {
    if (!replay) return;
    setIsPlaying(false);
    setCurrentFrameIndex((prev) => Math.min(prev + 1, replay.totalFrames - 1));
  }, [replay]);

  const stepBackward = useCallback(() => {
    setIsPlaying(false);
    setCurrentFrameIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleSetSpeed = useCallback((newSpeed: number) => {
    setSpeed(Math.max(0.1, Math.min(4, newSpeed)));
  }, []);

  // Current frame data
  const currentFrame = replay?.frames[currentFrameIndex] ?? null;

  return [
    {
      isPlaying,
      currentFrameIndex,
      totalFrames: replay?.totalFrames ?? 0,
      speed,
      currentFrame,
    },
    {
      play,
      pause,
      togglePlayPause,
      seek,
      stepForward,
      stepBackward,
      setSpeed: handleSetSpeed,
    },
  ];
}
