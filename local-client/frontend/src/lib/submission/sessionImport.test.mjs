import assert from "node:assert/strict";
import test from "node:test";

import { parseSessionNames, sessionNameError } from "./sessionImport.ts";

test("parses one session per line, trims blanks, and preserves order", () => {
  const parsed = parseSessionNames("  query-p1-11-kis  \n\nquery-p1-12-qa\n");

  assert.deepEqual(parsed, {
    valid: ["query-p1-11-kis", "query-p1-12-qa"],
    existing: [],
    duplicates: [],
    invalid: [],
  });
});

test("separates existing, repeated, and invalid session names", () => {
  const parsed = parseSessionNames(
    "new-session\nalready-there\nnew-session\nbad session\n../escape",
    ["already-there"],
  );

  assert.deepEqual(parsed, {
    valid: ["new-session"],
    existing: ["already-there"],
    duplicates: ["new-session"],
    invalid: ["bad session", "../escape"],
  });
});

test("uses the same 1-64 character session-name contract as the API", () => {
  assert.equal(sessionNameError("query_01"), null);
  assert.match(sessionNameError("-starts-wrong"), /letter or number/i);
  assert.match(sessionNameError(`a${"b".repeat(64)}`), /64 characters/i);
});
