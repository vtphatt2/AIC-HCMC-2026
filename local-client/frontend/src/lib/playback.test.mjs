import assert from "node:assert/strict";
import test from "node:test";

import {
  choosePlaybackSource,
  frameAtPlaybackTime,
  frameAtTimelineTime,
  normalizePlaybackFps,
  otherPlaybackSource,
  parseFrameTimeline,
  timeOfTimelineFrame,
} from "./playback.ts";

test("prefers YouTube initially when both playback sources are available", () => {
  assert.equal(choosePlaybackSource("youtube", true, true), "youtube");
  assert.equal(choosePlaybackSource("mp4", true, true), "mp4");
});

test("falls back to the remaining playback source", () => {
  assert.equal(choosePlaybackSource("youtube", false, true), "mp4");
  assert.equal(choosePlaybackSource("mp4", true, false), "youtube");
  assert.equal(choosePlaybackSource("youtube", false, false), null);
});

test("toggle selects the other available source", () => {
  assert.equal(otherPlaybackSource("youtube", true, true), "mp4");
  assert.equal(otherPlaybackSource("mp4", true, true), "youtube");
  assert.equal(otherPlaybackSource("mp4", false, true), null);
});

test("authoritative video metadata replaces a stale result fps", () => {
  assert.equal(normalizePlaybackFps(30, 25), 30);
  assert.equal(normalizePlaybackFps(undefined, 29.97), 29.97);
  assert.equal(normalizePlaybackFps(0, 0), 25);
});

test("frame number follows the active player's clock and refreshed fps", () => {
  assert.equal(frameAtPlaybackTime(10.5, 30), 315);
  assert.equal(frameAtPlaybackTime(10.5, 25), 262);
  assert.equal(frameAtPlaybackTime(-1, 30), 0);
});

test("an edited MOV seeks and reports the decoded frame by presentation time", () => {
  const buffer = new ArrayBuffer(5 * 4);
  const view = new DataView(buffer);
  [0, 40_000, 80_000, 5_080_000, 5_120_000].forEach((time, frame) =>
    view.setUint32(frame * 4, time, true));
  const timeline = parseFrameTimeline(buffer);
  assert.equal(timeOfTimelineFrame(timeline, 3), 5.08);
  assert.equal(timeOfTimelineFrame(timeline, 5), null);
  assert.equal(frameAtTimelineTime(timeline, 5.079), 2);
  assert.equal(frameAtTimelineTime(timeline, 5.08), 3);
  assert.equal(frameAtTimelineTime(timeline, 10), 4);
  assert.equal(frameAtPlaybackTime(5.08, 25), 127); // The old estimate was wrong.
});

test("invalid timeline data is rejected instead of silently mapping wrong frames", () => {
  assert.throws(() => parseFrameTimeline(new ArrayBuffer(3)));
  const buffer = new ArrayBuffer(12);
  const view = new DataView(buffer);
  [0, 20_000, 10_000].forEach((time, frame) => view.setUint32(frame * 4, time, true));
  assert.throws(() => parseFrameTimeline(buffer));
});

test('versioned timeline preserves source IDs across excluded timestamps', async () => {
  const { parseVersionedTimeline, submissionMilliseconds, frameAtSubmissionMilliseconds } = await import('./playback.ts');
  const timeline = parseVersionedTimeline({ version: 2, verified_timing: true, submission_unit: 'milliseconds',
    frame_ids: [0, 1, 4], source_pts: [200, 240, 320], presentation_us: [0, 40000, 120000],
    time_base: {num: 1, den: 1000}, playback_origin_pts: 200 });
  assert.equal(timeOfTimelineFrame(timeline, 2), null);
  assert.equal(timeOfTimelineFrame(timeline, 4), .12);
  assert.equal(frameAtTimelineTime(timeline, .121), 4);
  assert.equal(submissionMilliseconds(timeline, 4), 320);
  assert.equal(frameAtSubmissionMilliseconds(timeline, 318), 4);
  assert.throws(() => submissionMilliseconds(timeline, 2));
});
