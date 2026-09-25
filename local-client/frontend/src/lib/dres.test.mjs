import assert from "node:assert/strict";
import test from "node:test";

import { buildDresSubmission } from "./dres.ts";

test("KIS uses a video ID without extension and frame time in milliseconds", () => {
  assert.deepEqual(buildDresSubmission("task-1", "kis", { videoId: "L01_V001", frames: [42] }, 25), {
    answerSets: [{ taskId: "task-1", answers: [{ mediaItemName: "L01_V001", start: 1680, end: 1680 }] }],
  });
});

test("Q&A and TRAKE use the BTC text formats", () => {
  assert.deepEqual(buildDresSubmission("task-2", "qa", { videoId: "L01_V001", frames: [30], answer: "mango" }, 30), {
    answerSets: [{ taskId: "task-2", answers: [{ text: "QA-mango-L01_V001-1000" }] }],
  });
  assert.deepEqual(buildDresSubmission("task-3", "trake", { videoId: "L01_V001", frames: [42, 96] }), {
    answerSets: [{ taskId: "task-3", answers: [{ text: "TR-L01_V001-42,96" }] }],
  });
});

test("verified N positions are already source milliseconds", () => {
  const row = { videoId: "N001", frames: [1743], unit: "milliseconds", timingStatus: "verified", sourceFrames: [42] };
  assert.deepEqual(buildDresSubmission("task-n", "kis", row), {
    answerSets: [{ taskId: "task-n", answers: [{ mediaItemName: "N001", start: 1743, end: 1743 }] }],
  });
  assert.deepEqual(buildDresSubmission("task-n", "qa", { ...row, answer: "bus" }), {
    answerSets: [{ taskId: "task-n", answers: [{ text: "QA-bus-N001-1743" }] }],
  });
});

test("incomplete or invalid candidates cannot be submitted", () => {
  assert.throws(() => buildDresSubmission("task-1", "kis", { videoId: "L01_V001", frames: [42] }), /FPS/);
  assert.throws(() => buildDresSubmission("task-1", "qa", { videoId: "L01_V001", frames: [42], answer: "" }, 25), /answer/);
  assert.throws(() => buildDresSubmission("task-1", "trake", { videoId: "L01_V001", frames: [42, 42] }), /distinct/);
  assert.throws(() => buildDresSubmission("", "trake", { videoId: "L01_V001", frames: [42] }), /task/);
  assert.throws(() => buildDresSubmission("task-1", "kis", { videoId: "N001", frames: [42] }, 25), /source PTS/);
  assert.throws(() => buildDresSubmission("task-1", "trake", { videoId: "N001", frames: [42] }), /unavailable for TRAKE/);
  assert.throws(() => buildDresSubmission("task-1", "kis", { videoId: "L01_V001", frames: [42], unit: "milliseconds" }, 25), /unit/);
});
