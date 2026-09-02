import assert from "node:assert/strict";
import test from "node:test";

import {
  choosePlaybackSource,
  frameAtPlaybackTime,
  normalizePlaybackFps,
  otherPlaybackSource,
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
