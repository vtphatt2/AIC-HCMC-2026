export type PlaybackSource = "youtube" | "mp4";

export function choosePlaybackSource(
  preferred: PlaybackSource,
  youtubeAvailable: boolean,
  mp4Available: boolean,
): PlaybackSource | null {
  if (preferred === "youtube" && youtubeAvailable) return "youtube";
  if (preferred === "mp4" && mp4Available) return "mp4";
  if (youtubeAvailable) return "youtube";
  if (mp4Available) return "mp4";
  return null;
}

export function otherPlaybackSource(
  active: PlaybackSource,
  youtubeAvailable: boolean,
  mp4Available: boolean,
): PlaybackSource | null {
  if (active === "youtube" && mp4Available) return "mp4";
  if (active === "mp4" && youtubeAvailable) return "youtube";
  return null;
}

export function normalizePlaybackFps(
  authoritativeFps: number | null | undefined,
  resultFps: number | null | undefined,
): number {
  if (Number.isFinite(authoritativeFps) && Number(authoritativeFps) > 0) {
    return Number(authoritativeFps);
  }
  if (Number.isFinite(resultFps) && Number(resultFps) > 0) return Number(resultFps);
  return 25;
}

export function frameAtPlaybackTime(currentTimeSec: number, fps: number): number {
  if (!Number.isFinite(currentTimeSec) || currentTimeSec <= 0) return 0;
  return Math.floor(currentTimeSec * normalizePlaybackFps(fps, 25));
}
