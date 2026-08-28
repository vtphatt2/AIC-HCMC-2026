import assert from "node:assert/strict";
import test from "node:test";

import {
  buildTranscriptSearchPayload,
  highlightTranscriptText,
  transcriptChunksToFrameResults,
} from "./transcriptSearch.ts";

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

test("video mode keeps transcript hits that have no pre-resolved thumbnail", () => {
  const results = transcriptChunksToFrameResults([{
    chunk_id: 7,
    video_id: "L01_V001",
    youtube_id: "",
    topic: "Thời sự",
    start_time_ms: 1000,
    end_time_ms: 3000,
    text: "xin chào",
    score: 0.8,
    frame_image_url: "",
    frame_number: 0,
    nearest_timestamp_ms: null,
  }]);

  assert.equal(results.length, 1);
  assert.equal(results[0].timestamp_ms, 2000);
  assert.equal(results[0].frame_image_url, "/api/zip-frame/L01_V001/2000");
});

test("score mode highlights exact, accent-insensitive, and typo-similar query words", () => {
  const parts = highlightTranscriptText(
    "Cách nấu phở trong chương trình truyền hình",
    "nau pho truyenf ai",
  );
  const highlighted = parts.filter((part) => part.highlighted).map((part) => part.text);

  assert.deepEqual(highlighted, ["nấu", "phở", "truyền"]);
  assert.equal(parts.map((part) => part.text).join(""), "Cách nấu phở trong chương trình truyền hình");
});
