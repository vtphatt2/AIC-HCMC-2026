import assert from "node:assert/strict";
import test from "node:test";

import {
  COLLAPSED_CANDIDATE_LIMIT,
  visibleCandidateRows,
} from "./candidateVisibility.ts";

test("collapsed submission grids show at most the first five ranked candidates", () => {
  const rows = Array.from({ length: 100 }, (_, index) => ({ rank: index + 1 }));

  assert.equal(COLLAPSED_CANDIDATE_LIMIT, 5);
  assert.deepEqual(visibleCandidateRows(rows, false), rows.slice(0, 5));
});

test("expanded submission grids show every candidate", () => {
  const rows = Array.from({ length: 100 }, (_, index) => ({ rank: index + 1 }));

  assert.deepEqual(visibleCandidateRows(rows, true), rows);
});

test("small sessions remain unchanged when collapsed", () => {
  const rows = [{ rank: 1 }, { rank: 2 }, { rank: 3 }];

  assert.deepEqual(visibleCandidateRows(rows, false), rows);
  assert.deepEqual(rows, [{ rank: 1 }, { rank: 2 }, { rank: 3 }]);
});
