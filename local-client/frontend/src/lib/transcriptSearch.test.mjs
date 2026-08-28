import assert from "node:assert/strict";
import test from "node:test";

import { buildTranscriptSearchPayload } from "./transcriptSearch.ts";

test("the selected transcript algorithm is always sent explicitly", () => {
  assert.deepEqual(
    buildTranscriptSearchPayload("xin chào", 20, "Thời sự", "lexical"),
    {
      query: "xin chào",
      top_k: 20,
      topic_filter: "Thời sự",
      algorithm: "lexical",
    },
  );
});

test("topic is omitted when the selected algorithm does not support it", () => {
  assert.deepEqual(
    buildTranscriptSearchPayload("xin chào", 20, undefined, "fuzzy"),
    { query: "xin chào", top_k: 20, algorithm: "fuzzy" },
  );
});
