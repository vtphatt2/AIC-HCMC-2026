import os
import asyncpg

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.getenv("POSTGRES_URL"), min_size=2, max_size=10)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── DDL ──────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     VARCHAR(64)  PRIMARY KEY,
    title        TEXT         NOT NULL,
    youtube_id   VARCHAR(32)  NOT NULL,
    fps          FLOAT        NOT NULL DEFAULT 25.0,
    duration_ms  BIGINT       NOT NULL,
    frame_count  INT          NOT NULL,
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ocr_frames (
    id           SERIAL       PRIMARY KEY,
    frame_id     VARCHAR(128) NOT NULL UNIQUE,
    video_id     VARCHAR(64)  NOT NULL REFERENCES videos(video_id),
    frame_number INT          NOT NULL,
    timestamp_ms BIGINT       NOT NULL,
    ocr_text     TEXT         NOT NULL
);

-- Full-text search on OCR
CREATE INDEX IF NOT EXISTS idx_ocr_fts      ON ocr_frames USING GIN(to_tsvector('english', ocr_text));
CREATE INDEX IF NOT EXISTS idx_ocr_video    ON ocr_frames(video_id, frame_number);

CREATE TABLE IF NOT EXISTS transcripts (
    id            SERIAL      PRIMARY KEY,
    video_id      VARCHAR(64) NOT NULL REFERENCES videos(video_id),
    start_time_ms BIGINT      NOT NULL,
    end_time_ms   BIGINT      NOT NULL,
    text          TEXT        NOT NULL
);

-- Fast interval range queries: WHERE video_id = $1 AND start_time_ms <= $2 AND end_time_ms >= $3
CREATE INDEX IF NOT EXISTS idx_transcripts_video    ON transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_interval ON transcripts(video_id, start_time_ms, end_time_ms);
"""


async def init_schema():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)


# ── Query helpers ─────────────────────────────────────────────────────────────

async def fetch_video_metadata(video_ids: list[str]) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM videos WHERE video_id = ANY($1)", video_ids
    )
    return [dict(r) for r in rows]


async def fetch_ocr_by_frame_ids(frame_ids: list[str]) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM ocr_frames WHERE frame_id = ANY($1)", frame_ids
    )
    return [dict(r) for r in rows]


async def search_ocr_text(query: str, limit: int = 100) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT *, ts_rank(to_tsvector('english', ocr_text), plainto_tsquery('english', $1)) AS rank
        FROM ocr_frames
        WHERE to_tsvector('english', ocr_text) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
        """,
        query, limit,
    )
    return [dict(r) for r in rows]


async def search_transcript_text(query: str, limit: int = 100) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT *, ts_rank(to_tsvector('english', text), plainto_tsquery('english', $1)) AS rank
        FROM transcripts
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
        """,
        query, limit,
    )
    return [dict(r) for r in rows]


async def fetch_transcripts_in_range(video_id: str, start_ms: int, end_ms: int) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM transcripts
        WHERE video_id = $1 AND start_time_ms <= $2 AND end_time_ms >= $3
        ORDER BY start_time_ms
        """,
        video_id, end_ms, start_ms,
    )
    return [dict(r) for r in rows]
