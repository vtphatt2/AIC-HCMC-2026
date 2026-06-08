import asyncio
import logging
import os
import time
from app.db import milvus_client, postgres_client
from app.services.text_encoder import PECoreTextEncoder

ENV_MODE = os.getenv("ENV_MODE", "SERVER")
logger = logging.getLogger(__name__)


class DataProvider:
    """
    SERVER mode: reads directly from local Milvus and PostgreSQL using native SDKs.
    This gives maximum speed and zero network latency during competition.

    The interface is identical to the local-client DataProvider so strategy files
    can be copied to this server without changing a single line.
    """

    def __init__(self):
        if ENV_MODE != "SERVER":
            raise RuntimeError(
                f"remote-server DataProvider only supports ENV_MODE=SERVER, got '{ENV_MODE}'"
            )
        self._collection = milvus_client.get_collection()
        self._text_encoder = PECoreTextEncoder()

    def warmup_text_encoder(self, query: str = "warmup query") -> dict:
        return self._text_encoder.warmup(query)

    async def get_raw_data(self, query_groups: list[dict], limit: int = 1000) -> dict:
        """
        Fetch multi-modal raw data for all query groups from local databases.
        Returns a unified dict that strategies receive as `raw_data`.
        """
        all_frame_ids: set[str] = set()
        all_video_ids: set[str] = set()
        frames: list[dict] = []
        ocr: list[dict] = []
        transcripts: list[dict] = []

        for group_index, group in enumerate(query_groups):
            semantic_query = group.get("semantic_query", "").strip()
            text_query = group.get("text_query", "").strip()

            if semantic_query:
                query_vector = await asyncio.to_thread(self._text_encoder.encode, semantic_query)
                search_limit = max(int(limit), 1)
                timer_start = time.monotonic()
                hits = milvus_client.vector_search(self._collection, query_vector.tolist(), top_k=search_limit)
                logger.info(
                    "[TIMER] milvus_search %.3f ms group=%s top_k=%s hits=%s",
                    (time.monotonic() - timer_start) * 1000,
                    group_index,
                    search_limit,
                    len(hits),
                )
                for hit in hits:
                    hit["_query_group_index"] = group_index
                frames.extend(hits)
                all_frame_ids.update(h["frame_id"] for h in hits)
                all_video_ids.update(h["video_id"] for h in hits)
                logger.info("Milvus semantic query group=%s returned %s hits", group_index, len(hits))

            # OCR / transcript text search
            if text_query:
                ocr_hits = await postgres_client.search_ocr_text(text_query, limit=limit)
                for hit in ocr_hits:
                    hit["_query_group_index"] = group_index
                ocr.extend(ocr_hits)
                all_frame_ids.update(h["frame_id"] for h in ocr_hits)
                all_video_ids.update(h["video_id"] for h in ocr_hits)

                transcript_hits = await postgres_client.search_transcript_text(text_query, limit=limit)
                for hit in transcript_hits:
                    hit["_query_group_index"] = group_index
                transcripts.extend(transcript_hits)
                all_video_ids.update(h["video_id"] for h in transcript_hits)
                frames.extend(self._frames_for_transcripts(transcript_hits, limit=limit))
                logger.info(
                    "Postgres text query group=%s returned %s OCR hits and %s transcript hits",
                    group_index,
                    len(ocr_hits),
                    len(transcript_hits),
                )

        # Fetch video metadata for all referenced videos
        video_rows = await postgres_client.fetch_video_metadata(list(all_video_ids))
        videos = {v["video_id"]: dict(v) for v in video_rows}

        # DataProvider applies limit per query group above. Keep all groups here
        # so temporal strategies do not lose later-step candidates.
        return {
            "frames":      _dedupe_frames(frames),
            "ocr":         _dedupe_by_key(ocr, "frame_id"),
            "transcripts": transcripts,
            "videos":      videos,
        }

    def _frames_for_transcripts(self, transcript_hits: list[dict], limit: int) -> list[dict]:
        frames = []
        if not transcript_hits:
            return frames

        per_interval_limit = max(1, min(20, limit // max(1, len(transcript_hits))))
        for hit in transcript_hits:
            interval_frames = milvus_client.query_frames_in_time_range(
                self._collection,
                hit["video_id"],
                int(hit["start_time_ms"]),
                int(hit["end_time_ms"]),
                limit=per_interval_limit,
            )
            frames.extend(interval_frames)
            if len(frames) >= limit:
                break
        return frames[:limit]


def _dedupe_by_key(rows: list[dict], key: str) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(row)
    return deduped


def _dedupe_frames(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        identity = (row.get("frame_id"), row.get("_query_group_index"))
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped
