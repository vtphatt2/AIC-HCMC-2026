import assert from "node:assert/strict";
import test from "node:test";

import { mergeFetchedSubmissionStates } from "./mergeStates.ts";

const oldState = { session: "old", rows: [{ videoId: "L01_V001" }] };
const freshState = { session: "fresh", rows: [{ videoId: "L02_V001" }] };

test("keeps only valid states when a first fetch fails", () => {
  const merged = mergeFetchedSubmissionStates(
    [{ session: "fresh" }, { session: "missing" }],
    [freshState, null],
    {},
  );

  assert.deepEqual(merged, { fresh: freshState });
  assert.ok(Object.values(merged).every((state) => Array.isArray(state.rows)));
});

test("retains the previous state when a refresh fetch fails", () => {
  const merged = mergeFetchedSubmissionStates(
    [{ session: "old" }],
    [null],
    { old: oldState },
  );

  assert.deepEqual(merged, { old: oldState });
});
