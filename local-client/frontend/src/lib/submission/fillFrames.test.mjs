import assert from "node:assert/strict";
import test from "node:test";

import { fillKisRows } from "./fillFrames.ts";

test("keeps ranked seeds first and distributes 15-frame neighbors round-robin", () => {
  const seeds = [
    { videoId: "L01_V001", frames: [100] },
    { videoId: "L02_V002", frames: [200] },
    { videoId: "L03_V003", frames: [300] },
  ];

  assert.deepEqual(fillKisRows(seeds, 9, 15), [
    ...seeds,
    { videoId: "L01_V001", frames: [85] },
    { videoId: "L02_V002", frames: [185] },
    { videoId: "L03_V003", frames: [285] },
    { videoId: "L01_V001", frames: [115] },
    { videoId: "L02_V002", frames: [215] },
    { videoId: "L03_V003", frames: [315] },
  ]);
});

test("skips negative and duplicate video/frame pairs", () => {
  const rows = [
    { videoId: "L01_V001", frames: [5] },
    { videoId: "L01_V001", frames: [20] },
  ];
  const filled = fillKisRows(rows, 6, 15);

  assert.deepEqual(filled, [
    ...rows,
    { videoId: "L01_V001", frames: [35] },
    { videoId: "L01_V001", frames: [50] },
    { videoId: "L01_V001", frames: [65] },
    { videoId: "L01_V001", frames: [80] },
  ]);
  assert.ok(filled.every((row) => row.frames[0] >= 0));
  assert.equal(new Set(filled.map((row) => `${row.videoId}:${row.frames[0]}`)).size, filled.length);
});

test("does not mutate inputs, trim existing rows, or invent rows without seeds", () => {
  const rows = [{ videoId: "L01_V001", frames: [100] }];
  const snapshot = structuredClone(rows);

  assert.deepEqual(fillKisRows(rows, 1, 15), rows);
  assert.deepEqual(fillKisRows(rows, 0, 15), rows);
  assert.deepEqual(fillKisRows([], 100, 15), []);
  assert.deepEqual(rows, snapshot);
});

test("rejects invalid filler settings", () => {
  assert.throws(() => fillKisRows([{ videoId: "L01_V001", frames: [1] }], 100, 0), /step/i);
  assert.throws(() => fillKisRows([{ videoId: "L01_V001", frames: [1] }], 1.5, 15), /target/i);
});
