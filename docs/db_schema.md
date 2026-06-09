# Database Schema

The system uses two databases in production:
- **PostgreSQL** — structured metadata: video info, per-frame OCR text, transcript intervals
- **Milvus** — vector database: per-frame visual embeddings (1280-dim)

In `ENV_MODE=MOCK`, both are replaced by JSON files in
`local-client/local-backend/app/mock/`. In `ENV_MODE=SAMPLE`, the local backend
reads metadata/keyframes and performs linear search over local PE-Core vectors.

---

## PostgreSQL

### `videos`

One row per video.

```sql
CREATE TABLE videos (
    video_id     VARCHAR(64)  PRIMARY KEY,   -- internal dataset ID, e.g. "L01_V001"
    title        TEXT         NOT NULL,
    youtube_id   VARCHAR(32)  NOT NULL,      -- YouTube ID, e.g. "JbYI8OEkFK4"
    fps          FLOAT        NOT NULL DEFAULT 25.0,
    duration_ms  BIGINT       NOT NULL,      -- total duration in milliseconds
    frame_count  INT          NOT NULL,      -- total extracted frames
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);
```

### `ocr_frames`

One row per frame that contains readable text. Not every frame has an OCR row.

```sql
CREATE TABLE ocr_frames (
    id           SERIAL       PRIMARY KEY,
    frame_id     VARCHAR(128) NOT NULL UNIQUE,  -- e.g. "L01_V001_000022"
    video_id     VARCHAR(64)  NOT NULL REFERENCES videos(video_id),
    frame_number INT          NOT NULL,
    timestamp_ms BIGINT       NOT NULL,
    ocr_text     TEXT         NOT NULL
);

-- Full-text search using PostgreSQL GIN index
CREATE INDEX idx_ocr_fts   ON ocr_frames USING GIN(to_tsvector('english', ocr_text));
-- Fast lookup by video + frame
CREATE INDEX idx_ocr_video ON ocr_frames(video_id, frame_number);
```

**Frame ID convention:** `{video_id}_{frame_number:06d}` — zero-padded to 6 digits.

Example query — find frames containing "GOAL":
```sql
SELECT * FROM ocr_frames
WHERE to_tsvector('english', ocr_text) @@ plainto_tsquery('english', 'GOAL')
ORDER BY ts_rank(to_tsvector('english', ocr_text), plainto_tsquery('english', 'GOAL')) DESC
LIMIT 100;
```

### `transcripts`

One row per speech segment. A segment spans a time range, not a single frame.

```sql
CREATE TABLE transcripts (
    id            SERIAL      PRIMARY KEY,
    video_id      VARCHAR(64) NOT NULL REFERENCES videos(video_id),
    start_time_ms BIGINT      NOT NULL,
    end_time_ms   BIGINT      NOT NULL,
    text          TEXT        NOT NULL
);

-- Fast interval range queries
CREATE INDEX idx_transcripts_video    ON transcripts(video_id);
CREATE INDEX idx_transcripts_interval ON transcripts(video_id, start_time_ms, end_time_ms);
```

Example query — find all speech overlapping a given timestamp T:
```sql
SELECT * FROM transcripts
WHERE video_id = 'L01_V001'
  AND start_time_ms <= 5000    -- T (ms)
  AND end_time_ms   >= 5000    -- T (ms)
ORDER BY start_time_ms;
```

### Converting between frame number and timestamp

```python
timestamp_ms = int(frame_number / fps * 1000)
frame_number = int(timestamp_ms / 1000 * fps)

# seek time for YouTube IFrame API
seek_seconds = timestamp_ms / 1000
```

---

## Milvus

### Collection: `video_frames`

```python
fields = [
    FieldSchema(name="frame_id",     dtype=VARCHAR,       max_length=128, is_primary=True),
    FieldSchema(name="video_id",     dtype=VARCHAR,       max_length=64),
    FieldSchema(name="frame_number", dtype=INT64),
    FieldSchema(name="timestamp_ms", dtype=INT64),
    FieldSchema(name="image_url",    dtype=VARCHAR,       max_length=256),
    FieldSchema(name="vector",       dtype=FLOAT_VECTOR,  dim=1280),
]
```

**Embedding model:** `timm/PE-Core-bigG-14-448` → 1280-dimensional float vectors.

**Index:** HNSW with COSINE metric.

```python
index_params = {
    "metric_type":  "COSINE",
    "index_type":   "HNSW",
    "params": {
        "M":              16,    # number of bi-directional links per node
        "efConstruction": 256,   # size of the dynamic candidate list during build
    }
}
```

**Search parameters:**
```python
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": max(256, top_k)}
}
```

The backend caps `top_k` at 1000 and keeps `ef >= top_k`. COSINE similarity
scores range from -1 to 1. Higher is more similar. Results are returned sorted
descending by score.

### Relationship between Milvus and PostgreSQL

- Every embedded frame has one row in Milvus (`video_frames`)
- Only frames with visible text have a row in `ocr_frames`
- `frame_id` is the join key between the two systems
- `videos` is the authoritative source of `fps`, `duration_ms`, and `youtube_id`
- `video_id` identifies dataset records; `youtube_id` is used only for YouTube playback

```
Milvus: video_frames.frame_id  ──►  PostgreSQL: ocr_frames.frame_id
                                                videos.video_id
```

---

## Mock Data (Development)

Located in `local-client/local-backend/app/mock/`. These files mirror the shape of the production databases.

| File | Rows | Equivalent to |
|---|---|---|
| `mock_videos.json` | 3 | `videos` table |
| `mock_frames.json` | 105 | Milvus `video_frames` collection (no vectors) |
| `mock_ocr.json` | 10 | `ocr_frames` table |
| `mock_transcripts.json` | 11 | `transcripts` table |

Mock frames do **not** contain embedding vectors — the example strategy uses random scores instead. Frame images use `https://picsum.photos/seed/{frame_id}/320/180` as placeholders; replace with real frame paths when the ingestion pipeline is ready.
