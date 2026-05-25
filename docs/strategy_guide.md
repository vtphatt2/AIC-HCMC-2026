# Strategy Developer Guide

This guide explains how to write, test, and promote a retrieval strategy.

---

## What is a Strategy?

A strategy is a single Python file in `app/strategies/` that implements one core method: **`fusion_and_temporal()`**. It receives raw multi-modal data that has already been fetched from the database and returns a ranked list of frames.

The engine handles everything else: fetching data, enforcing timeouts, applying top-K, and serving results to the frontend.

---

## Quickstart — Create Your First Strategy

```bash
# 1. Copy the example file
cp local-client/local-backend/app/strategies/example_strategy.py \
   local-client/local-backend/app/strategies/yourname_v1.py

# 2. Edit the file (see below)
# 3. Restart the backend — your strategy appears in the dropdown automatically
```

---

## Strategy File Template

```python
from app.strategies.base_strategy import BaseStrategy


class YournameV1(BaseStrategy):
    # ── Required metadata ─────────────────────────────────────────────────
    name        = "My Strategy v1"
    description = "One sentence explaining what this strategy does differently."
    author      = "Your Name"
    version     = "1.0"
    # ─────────────────────────────────────────────────────────────────────

    def fusion_and_temporal(self, raw_data: dict, query_groups: list[dict]) -> list[dict]:
        frames      = raw_data["frames"]       # list of frame dicts
        ocr         = raw_data["ocr"]          # list of OCR dicts
        transcripts = raw_data["transcripts"]  # list of transcript interval dicts
        videos      = raw_data["videos"]       # dict keyed by video_id

        results = []

        for frame in frames:
            video = videos.get(frame["video_id"], {})
            fps   = float(video.get("fps", 25.0))

            score = self._compute_score(frame, ocr, transcripts, query_groups)

            results.append({
                "video_id":        frame["video_id"],
                "frame_id":        frame["frame_id"],
                "frame_number":    frame["frame_number"],
                "timestamp_ms":    frame["timestamp_ms"],
                "confidence":      round(score, 4),    # float 0.0–1.0
                "frame_image_url": frame["image_url"],
                "fps":             fps,
            })

        # Sort descending by confidence before returning
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _compute_score(self, frame, ocr, transcripts, query_groups):
        # Your scoring logic here
        return 0.5
```

---

## The `raw_data` Dict

Your `fusion_and_temporal()` always receives the same structure regardless of whether the backend is in MOCK or LOCAL or SERVER mode.

### `raw_data["frames"]` — list of dicts

```python
{
    "frame_id":     "dQw4w9WgXcQ_000025",  # unique ID
    "video_id":     "dQw4w9WgXcQ",
    "frame_number": 25,                    # raw frame index (0-based)
    "timestamp_ms": 1000,                  # frame_number / fps * 1000
    "image_url":    "https://..."          # URL to the frame image
}
```

### `raw_data["ocr"]` — list of dicts

Only frames that contain visible text appear here. Not every frame has an OCR entry.

```python
{
    "frame_id":     "dQw4w9WgXcQ_000025",
    "video_id":     "dQw4w9WgXcQ",
    "frame_number": 25,
    "timestamp_ms": 1000,
    "ocr_text":     "BREAKING NEWS: City Center Flooding"
}
```

### `raw_data["transcripts"]` — list of dicts

Speech intervals. One interval can span many frames.

```python
{
    "video_id":      "dQw4w9WgXcQ",
    "start_time_ms": 0,
    "end_time_ms":   4000,
    "text":          "Floodwaters have reached the city centre..."
}
```

### `raw_data["videos"]` — dict keyed by `video_id`

```python
{
    "dQw4w9WgXcQ": {
        "video_id":    "dQw4w9WgXcQ",
        "title":       "Mock Video — City Street Scene",
        "youtube_id":  "dQw4w9WgXcQ",
        "fps":         25.0,
        "duration_ms": 212000,
        "frame_count": 5300
    }
}
```

### `query_groups` — list of dicts

The search queries from the frontend, one dict per temporal step.

```python
[
    # Step 1 (always present)
    {
        "semantic_query":    "a dog running in the park",  # visual description
        "text_query":        "DOG PARK",                  # OCR / transcript keyword
        "temporal_offset_ms": 0                           # always 0 for the first step
    },
    # Step 2 (optional, added via "Add Temporal Step")
    {
        "semantic_query":    "a cat sitting",
        "text_query":        "",
        "temporal_offset_ms": 5000   # this scene occurs ~5s after step 1
    }
]
```

---

## Lifecycle Methods

You can override any of these. Only `fusion_and_temporal` is required.

```
search(query_groups)
  │
  ├─ pre_process(query_groups)         optional — transform queries before fetch
  │
  ├─ DataProvider.get_raw_data()       automatic — fetches data (capped at 1000)
  │
  ├─ fusion_and_temporal(raw_data,     REQUIRED — your scoring algorithm
  │                      query_groups)            runs in a thread, 2s timeout
  │
  └─ post_filter(results)              optional — filter or re-rank after scoring
```

### `pre_process` example — normalise queries

```python
def pre_process(self, query_groups):
    for group in query_groups:
        group["semantic_query"] = group["semantic_query"].lower().strip()
    return query_groups
```

### `post_filter` example — remove low-confidence results

```python
def post_filter(self, results):
    return [r for r in results if r["confidence"] >= 0.5]
```

---

## Guardrails

These are enforced automatically and cannot be disabled:

| Guardrail | Limit | What happens if exceeded |
|---|---|---|
| **Fetch cap** | 1000 records per query | DataProvider truncates silently |
| **Execution timeout** | 2.0 seconds | HTTP request returns 408 with error message |

The timeout uses `asyncio.wait_for` around a thread. The response returns immediately after 2 seconds but the thread may keep running briefly in the background. If you have an infinite loop, restart the backend.

---

## Temporal Search Patterns

When the user adds a Temporal Step, `query_groups` has two or more entries. Here is a pattern for implementing temporal logic:

```python
def fusion_and_temporal(self, raw_data, query_groups):
    if len(query_groups) < 2:
        return self._single_pass(raw_data, query_groups[0])

    # Step 1 candidates
    step1_results = self._score_frames(raw_data, query_groups[0])

    # Step 2 candidates
    step2_results = self._score_frames(raw_data, query_groups[1])
    offset_ms     = query_groups[1]["temporal_offset_ms"]
    tolerance_ms  = 3000  # ± 3 seconds

    # Find pairs where step2 follows step1 by ~offset_ms
    step2_by_video = {}
    for r in step2_results:
        step2_by_video.setdefault(r["video_id"], []).append(r)

    results = []
    for r1 in step1_results:
        candidates = step2_by_video.get(r1["video_id"], [])
        for r2 in candidates:
            time_gap = r2["timestamp_ms"] - r1["timestamp_ms"]
            if abs(time_gap - offset_ms) <= tolerance_ms:
                # Combine scores and return the anchor frame (step 1)
                combined_score = (r1["confidence"] + r2["confidence"]) / 2
                results.append({**r1, "confidence": round(combined_score, 4)})
                break

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
```

---

## Scoring Ideas

| Signal | How to use it |
|---|---|
| Visual similarity | Cosine similarity from Milvus (available in SERVER/LOCAL mode as `frame["score"]`) |
| OCR exact match | Check if `ocr_text` contains the text query |
| OCR fuzzy match | Use `difflib.SequenceMatcher` or `rapidfuzz` |
| Transcript overlap | Find transcripts where `start_time_ms <= frame_ms <= end_time_ms` |
| Temporal proximity | Boost frames near a high-scoring anchor frame |
| Multi-modal fusion | Weighted sum: `0.6 * visual_score + 0.3 * ocr_score + 0.1 * transcript_score` |

---

## Naming Convention

Use descriptive file names so teammates can identify strategies in the dropdown:

```
yourname_approach_v1.py     # e.g. duy_temporal_v1.py
yourname_approach_v2.py     # iterate in new files, keep v1 as baseline
```

Never overwrite a teammate's file. Create a new version instead.

---

## Promoting to the GPU Server

When your strategy scores better than the current production baseline:

1. Copy the file exactly as-is to `remote-server/app/strategies/`
2. Restart the remote server
3. Select your strategy in the frontend dropdown
4. Run the official evaluation

Do **not** change the code when copying — the whole point of the DataProvider abstraction is that the same file works in both environments.
