export type PlaybackSource = "youtube" | "mp4";

export function initialPlaybackSource(videoId: string, youtubeId: string): PlaybackSource {
  // Long S broadcasts can have different lead-ins on YouTube. Keep YouTube
  // available as an alternate, but do not apply source timestamps to it until
  // alignment has been verified.
  if (videoId.startsWith("S")) return "mp4";
  return youtubeId ? "youtube" : "mp4";
}

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

/** Complete decoded-frame presentation timeline returned by the backend. */
export interface FrameTimeline {
  readonly view: DataView;
  readonly length: number;
  readonly frameIds?: number[];
  readonly sourcePts?: number[];
  readonly presentationUs?: number[];
  readonly timebase?: number;
}

export function parseFrameTimeline(buffer: ArrayBuffer): FrameTimeline {
  if (buffer.byteLength < 8 || buffer.byteLength % 4 !== 0) {
    throw new Error("Invalid frame timeline length");
  }
  const view = new DataView(buffer);
  const length = buffer.byteLength / 4;
  if (view.getUint32(0, true) !== 0) throw new Error("Frame timeline must start at zero");
  for (let frame = 1, previous = 0; frame < length; frame++) {
    const current = view.getUint32(frame * 4, true);
    if (current < previous) throw new Error("Frame timeline is not ordered");
    previous = current;
  }
  return { view, length };
}

export function timeOfTimelineFrame(timeline: FrameTimeline, frame: number): number | null {
  const position = timelinePosition(timeline, frame);
  return position < 0 ? null : timelineMicroseconds(timeline, position) / 1_000_000;
}

export function frameAtTimelineTime(timeline: FrameTimeline, seconds: number): number {
  if (!Number.isFinite(seconds) || seconds <= 0) return timeline.frameIds?.[0] ?? 0;
  const microseconds = seconds * 1_000_000;
  let lo = 0;
  let hi = timeline.length;
  while (lo < hi) {
    const middle = Math.floor((lo + hi) / 2);
    if (timelineMicroseconds(timeline, middle) <= microseconds) lo = middle + 1;
    else hi = middle;
  }
  return timeline.frameIds?.[Math.max(0, lo - 1)] ?? Math.max(0, lo - 1);
}

export function parseVersionedTimeline(payload: any): FrameTimeline {
  const { frame_ids: ids, source_pts: pts, presentation_us: times, time_base: base } = payload;
  if (payload.version !== 2 || payload.verified_timing !== true || payload.submission_unit !== 'milliseconds' ||
      !Array.isArray(ids) || !ids.length || !Array.isArray(pts) || !Array.isArray(times) ||
      ids.length !== pts.length || ids.length !== times.length || base?.num !== 1 ||
      !Number.isSafeInteger(base.den) || base.den <= 0 || !Number.isSafeInteger(payload.playback_origin_pts)) {
    throw new Error('Invalid versioned timeline');
  }
  for (let i = 0; i < ids.length; i++) {
    if (!Number.isSafeInteger(ids[i]) || ids[i] < 0 || !Number.isSafeInteger(pts[i]) ||
        !Number.isSafeInteger(times[i]) || times[i] < 0 ||
        (i > 0 && (ids[i] <= ids[i-1] || pts[i] <= pts[i-1] || times[i] <= times[i-1])) ||
        Math.abs(times[i] - (pts[i] - payload.playback_origin_pts) * 1_000_000 / base.den) >= 1.01) {
      throw new Error('Invalid source frame identity or timing');
    }
  }
  if (times[0] !== 0) throw new Error('Timeline must start at zero');
  return { view: new DataView(new ArrayBuffer(0)), length: ids.length,
           frameIds: ids, sourcePts: pts, presentationUs: times, timebase: base.den };
}

function timelinePosition(timeline: FrameTimeline, frame: number): number {
  if (!Number.isSafeInteger(frame) || frame < 0) return -1;
  if (!timeline.frameIds) return frame < timeline.length ? frame : -1;
  let lo = 0, hi = timeline.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (timeline.frameIds[mid] < frame) lo = mid + 1; else hi = mid;
  }
  return timeline.frameIds[lo] === frame ? lo : -1;
}

function timelineMicroseconds(timeline: FrameTimeline, position: number): number {
  return timeline.presentationUs?.[position] ?? timeline.view.getUint32(position * 4, true);
}

export function submissionMilliseconds(timeline: FrameTimeline, frame: number): number {
  const position = timelinePosition(timeline, frame);
  if (position < 0 || !timeline.sourcePts || !timeline.timebase) {
    throw new Error('Verified source timing is required for N submission');
  }
  const ms = Math.round(timeline.sourcePts[position] * 1000 / timeline.timebase);
  if (!Number.isSafeInteger(ms) || ms < 0) throw new Error('Invalid source submission time');
  return ms;
}

export function frameAtSubmissionMilliseconds(timeline: FrameTimeline, milliseconds: number): number {
  if (!timeline.sourcePts || !timeline.timebase || !timeline.frameIds) throw new Error('Verified source timeline required');
  if (!Number.isSafeInteger(milliseconds) || milliseconds < 0) throw new Error('Invalid source submission time');
  let lo = 0, hi = timeline.length;
  const target = milliseconds * timeline.timebase / 1000;
  // CSV carries rounded milliseconds. The first/last frame can therefore
  // round just outside the unrounded source clock and must still round-trip.
  const minimum = submissionMilliseconds(timeline, timeline.frameIds[0]);
  const maximum = submissionMilliseconds(timeline, timeline.frameIds[timeline.length - 1]);
  if (milliseconds < minimum || milliseconds > maximum) throw new Error('Position outside source timeline');
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (timeline.sourcePts[mid] < target) lo = mid + 1; else hi = mid;
  }
  const right = Math.min(lo, timeline.length - 1), left = Math.max(0, right - 1);
  return timeline.frameIds[target - timeline.sourcePts[left] <= timeline.sourcePts[right] - target ? left : right];
}
