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
    genre        VARCHAR(64),
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

CREATE TABLE IF NOT EXISTS transcript_chunks_metadata (
    chunk_id      BIGINT       PRIMARY KEY,
    video_id      VARCHAR(64)  NOT NULL,
    topic         VARCHAR(64)  NOT NULL,
    start_time_ms BIGINT       NOT NULL,
    end_time_ms   BIGINT       NOT NULL,
    raw_text      TEXT         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_video    ON transcript_chunks_metadata(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_topic    ON transcript_chunks_metadata(topic);
CREATE INDEX IF NOT EXISTS idx_chunks_interval ON transcript_chunks_metadata(video_id, start_time_ms, end_time_ms);
CREATE INDEX IF NOT EXISTS idx_chunks_fts      ON transcript_chunks_metadata
USING GIN(to_tsvector('simple', raw_text));
"""


async def init_schema():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)
        # Migration: add genre column if missing (safe to call repeatedly)
        await conn.execute(
            "ALTER TABLE videos ADD COLUMN IF NOT EXISTS genre VARCHAR(64)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_videos_genre ON videos(genre)"
        )


# ── Query helpers ─────────────────────────────────────────────────────────────

async def fetch_video_metadata(video_ids: list[str]) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM videos WHERE video_id = ANY($1)", video_ids
    )
    return [dict(r) for r in rows]


async def fetch_all_video_metadata() -> list[dict]:
    """Small catalog used by the title/ID autocomplete (one row per video)."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT video_id, title FROM videos ORDER BY video_id"
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


# ── Transcript Chunks Metadata ─────────────────────────────────────────────────

async def upsert_transcript_chunks(records: list[dict]) -> int:
    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO transcript_chunks_metadata (chunk_id, video_id, topic, start_time_ms, end_time_ms, raw_text)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (chunk_id) DO UPDATE SET
                video_id = EXCLUDED.video_id,
                topic = EXCLUDED.topic,
                start_time_ms = EXCLUDED.start_time_ms,
                end_time_ms = EXCLUDED.end_time_ms,
                raw_text = EXCLUDED.raw_text
            """,
            [
                (
                    int(r["chunk_id"]),
                    str(r["video_id"]),
                    str(r["topic"]),
                    int(r["start_time_ms"]),
                    int(r["end_time_ms"]),
                    str(r["raw_text"]),
                )
                for r in records
            ],
        )
        inserted = len(records)
    return inserted


async def fetch_transcript_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT chunk_id, video_id, topic, start_time_ms, end_time_ms, raw_text "
        "FROM transcript_chunks_metadata WHERE chunk_id = ANY($1)",
        chunk_ids,
    )
    return [dict(r) for r in rows]


async def search_transcript_chunks_text(
    query: str,
    limit: int = 100,
    video_genre: str = "All",
    topic_filter: str = "",
) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT c.chunk_id, c.video_id, c.topic, c.start_time_ms, c.end_time_ms,
               c.raw_text AS text,
               ts_rank(to_tsvector('simple', c.raw_text), plainto_tsquery('simple', $1)) AS score
        FROM transcript_chunks_metadata c
        LEFT JOIN videos v ON v.video_id = c.video_id
        WHERE to_tsvector('simple', c.raw_text) @@ plainto_tsquery('simple', $1)
          AND ($3 = 'All' OR $3 = '' OR v.genre = $3)
          AND ($4 = '' OR c.topic = $4)
        ORDER BY score DESC
        LIMIT $2
        """,
        query,
        limit,
        video_genre,
        topic_filter,
    )
    return [dict(row) for row in rows]


async def fetch_all_transcript_chunks(topic_filter: str | None = None) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT chunk_id, video_id, topic, start_time_ms, end_time_ms, raw_text
        FROM transcript_chunks_metadata
        WHERE ($1 = '' OR topic = $1)
        ORDER BY chunk_id
        """,
        topic_filter or "",
    )
    return [dict(row) for row in rows]


async def clear_transcript_chunks_for_video(video_id: str) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM transcript_chunks_metadata WHERE video_id = $1", video_id
        )
    return int(result.split()[-1]) if result else 0


# ── Video Genre ───────────────────────────────────────────────────────────────

async def fetch_video_ids_by_genre(genre: str) -> list[str]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT video_id FROM videos WHERE genre = $1", genre
    )
    return [r["video_id"] for r in rows]


async def upsert_video_metadata(videos: list[dict]) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO videos (video_id, title, youtube_id, fps, duration_ms, frame_count)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (video_id) DO UPDATE SET
                title = EXCLUDED.title,
                youtube_id = EXCLUDED.youtube_id,
                fps = EXCLUDED.fps,
                duration_ms = EXCLUDED.duration_ms,
                frame_count = EXCLUDED.frame_count
            """,
            [
                (
                    v["video_id"],
                    v.get("title") or v["video_id"],
                    v.get("youtube_id") or "",
                    float(v.get("fps") or 25.0),
                    int(v.get("duration_ms") or 0),
                    int(v.get("frame_count") or 0),
                )
                for v in videos
            ],
        )
    return len(videos)


async def update_video_genres(genre_map: dict[str, str]) -> int:
    """Bulk-update genre for multiple videos. genre_map: {video_id: genre}."""
    pool = await get_pool()
    updated = 0
    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE videos SET genre = $2 WHERE video_id = $1",
            [(vid, genre) for vid, genre in genre_map.items()],
        )
        updated = len(genre_map)
    return updated
