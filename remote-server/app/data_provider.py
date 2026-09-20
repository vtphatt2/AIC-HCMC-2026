import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.db import milvus_client, postgres_client
from app.services.blocking_io import BlockingIO
from app.services.text_encoder import PECoreTextEncoder
from app.services.transcript_search import TranscriptSearchService

ENV_MODE = os.getenv("ENV_MODE", "SERVER")
DEFAULT_VECTOR_SEARCH_ALGORITHM = {
    "milvus": "hnsw",
    "hnsw": "hnsw",
    "flat": "flat",
    "cagra": "cagra",
    "scann": "scann",
}.get(os.getenv("VECTOR_SEARCH_BACKEND", "milvus").lower(), "hnsw")
VALID_VECTOR_SEARCH_ALGORITHMS = {"hnsw", "flat", "cagra", "scann"}
TRANSCRIPT_CHUNK_SEARCH_ENABLED = os.getenv("TRANSCRIPT_CHUNK_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes"}
logger = logging.getLogger(__name__)
CHANNELS = {
    "raw.semantic",
    "subtitled.semantic",
    "transcript.lexical",
    "transcript.semantic",
}


def _exclude_frames_expr(frame_ids: list[str]) -> str | None:
    if not frame_ids:
        return None
    quoted = ", ".join(json.dumps(frame_id) for frame_id in frame_ids)
    return f"frame_id not in [{quoted}]"


def _combine_expr(*parts: str | None) -> str | None:
    values = [f"({part})" for part in parts if part]
    return " and ".join(values) or None


class DataProvider:
    """
    SERVER mode: reads vectors from Milvus or CAGRA and metadata from PostgreSQL.
    All services run locally to avoid network latency during competition.

    The interface is identical to the local-client DataProvider so strategy files
    can be copied to this server without changing a single line.
    """

    def __init__(self):
        if ENV_MODE != "SERVER":
            raise RuntimeError(
                f"remote-server DataProvider only supports ENV_MODE=SERVER, got '{ENV_MODE}'"
            )
        self._collections = {}
        self._db_io = BlockingIO(int(os.getenv("DB_IO_WORKERS", "2")))
        self._text_encoder = PECoreTextEncoder()
        self._text_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="text-encoder",
        )
        self._cagra = None
        self._cagra_lock = threading.Lock()
        if DEFAULT_VECTOR_SEARCH_ALGORITHM == "cagra":
            self._cagra = self._get_cagra()
        elif DEFAULT_VECTOR_SEARCH_ALGORITHM not in {"hnsw", "flat", "scann"}:
            raise ValueError("VECTOR_SEARCH_BACKEND must be 'milvus', 'hnsw', 'flat', 'cagra', or 'scann'")
        else:
            self._get_collection(DEFAULT_VECTOR_SEARCH_ALGORITHM)

        self._transcript_search: TranscriptSearchService | None = None
        if TRANSCRIPT_CHUNK_SEARCH_ENABLED:
            try:
                from pathlib import Path
                # Lot archives ship no JPGs, so this is usually an empty
                # directory and thumbnails resolve through /api/zip-frame.
                keyframe_dir = Path(
                    os.getenv("FRAME_STATIC_DIR", "")
                    or Path(__file__).resolve().parents[2] / "static" / "frames"
                )
                self._transcript_search = TranscriptSearchService(keyframe_dir=keyframe_dir)
                logger.info("Transcript chunk search enabled (keyframe_dir=%s)", keyframe_dir)
            except Exception as exc:
                logger.warning("Transcript chunk search unavailable: %s", exc)
                self._transcript_search = None

    async def warmup_text_encoder(self, query: str = "warmup query", passes: int = 10) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._text_executor,
            self._text_encoder.warmup,
            query,
            passes,
        )

    async def _encode_text(self, text: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._text_executor,
            self._text_encoder.encode,
            text,
        )

    def close(self) -> None:
        self._text_executor.shutdown(wait=False)
        self._db_io.close()

    async def _run_db(self, function, *args, **kwargs):
        return await self._db_io.run(function, *args, **kwargs)

    def _get_cagra(self):
        if self._cagra is not None:
            return self._cagra
        with self._cagra_lock:
            if self._cagra is None:
                from app.db.cagra_client import CagraClient

                self._cagra = CagraClient()
        return self._cagra

    def _get_collection(self, algorithm: str, channel: str = "raw.semantic"):
        algorithm = milvus_client.normalize_algorithm(algorithm)
        cache_key = (channel, algorithm)
        if cache_key not in self._collections:
            collection_name = milvus_client.collection_name_for_algorithm(algorithm, channel)
            if not milvus_client.has_collection_for_algorithm(algorithm, channel):
                raise ValueError(
                    f"{channel} collection '{collection_name}' is not available. "
                    f"Build it with --channel {channel} --vector-index {algorithm}."
                )
            self._collections[cache_key] = milvus_client.get_collection_for_name(collection_name)
        return self._collections[cache_key]

    def _metadata_collection(self):
        algorithm = DEFAULT_VECTOR_SEARCH_ALGORITHM
        if algorithm == "cagra":
            algorithm = "hnsw"
        return self._get_collection(algorithm)

    async def retrieve(
        self,
        channel: str,
        query: str,
        *,
        top_k: int = 100,
        video_genre: str = "All",
        vector_search_algorithm: str | None = None,
        exclude_frame_ids: list[str] | None = None,
        include_vector: bool = True,
    ) -> list[dict]:
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel '{channel}'. Available: {sorted(CHANNELS)}")
        query = str(query).strip()
        if not query:
            return []
        top_k = min(max(int(top_k), 1), 1000)
        excluded = list(dict.fromkeys(str(value) for value in (exclude_frame_ids or []) if value))

        if channel in {"raw.semantic", "subtitled.semantic"}:
            vector = await self._encode_text(query)
            algorithm = (
                str(vector_search_algorithm).strip().lower()
                if vector_search_algorithm
                else DEFAULT_VECTOR_SEARCH_ALGORITHM
            )
            if algorithm != "cagra":
                algorithm = milvus_client.normalize_algorithm(algorithm)
            search_expr = _combine_expr(
                await self._genre_expr(video_genre),
                _exclude_frames_expr(excluded),
            )
            if algorithm == "cagra":
                if channel != "raw.semantic":
                    raise ValueError("CAGRA is only indexed for raw.semantic")
                hits = self._get_cagra().search(
                    vector, top_k=min(1000, top_k + len(excluded))
                )
                excluded_set = set(excluded)
                hits = [hit for hit in hits if hit.get("frame_id") not in excluded_set][:top_k]
            else:
                hits = await self._run_db(milvus_client.vector_search,
                    self._get_collection(algorithm, channel),
                    vector.tolist(),
                    top_k=top_k,
                    algorithm=algorithm,
                    expr=search_expr,
                    # Deferred fetching uses the default raw collection below.
                    include_vector=(include_vector or channel != "raw.semantic"
                                    or algorithm != DEFAULT_VECTOR_SEARCH_ALGORITHM),
                )
            await self._hydrate_frames(hits)
            return [
                {**hit, "channel": channel, "item_id": hit["frame_id"], "rank": rank}
                for rank, hit in enumerate(hits, start=1)
            ]

        if channel == "transcript.lexical":
            hits = await postgres_client.search_transcript_chunks_text(query, top_k, video_genre)
            return [
                {**hit, "channel": channel, "rank": rank}
                for rank, hit in enumerate(hits, start=1)
            ]

        if self._transcript_search is None:
            raise RuntimeError("transcript.semantic is not configured")
        hits = await self._transcript_search.search_chunks(query, top_k=top_k)
        if video_genre and video_genre != "All":
            allowed = set(await postgres_client.fetch_video_ids_by_genre(video_genre))
            hits = [hit for hit in hits if hit["video_id"] in allowed]
        return hits

    async def frame_embeddings(self, frame_ids: list[str]) -> dict[str, list[float]]:
        frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids if frame_id))
        collection = self._metadata_collection()
        embeddings = {}
        for start in range(0, len(frame_ids), 1000):
            embeddings.update(await self._run_db(milvus_client.query_frame_vectors,
                collection, frame_ids[start:start + 1000]
            ))
        return embeddings

    async def keyframes(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        *,
        limit: int = 20,
    ) -> list[dict]:
        if int(start_ms) > int(end_ms):
            raise ValueError("start_ms must be <= end_ms")
        hits = await self._run_db(milvus_client.query_frames_in_time_range,
            self._metadata_collection(),
            video_id,
            int(start_ms),
            int(end_ms),
            limit=min(max(int(limit), 1), 1000),
        )
        await self._hydrate_frames(hits)
        return hits

    def results(self, hits: list[dict]) -> list[dict]:
        output = []
        seen = set()
        for hit in hits:
            frame_id = hit.get("frame_id")
            if not frame_id:
                raise ValueError("Final result hit must contain frame_id; call keyframes() for transcript chunks")
            if frame_id in seen:
                continue
            seen.add(frame_id)
            confidence = float(hit.get("confidence", hit.get("score", 0.0)))
            row = {
                **hit,
                "video_id": str(hit.get("video_id", "")),
                "youtube_id": str(hit.get("youtube_id", "")),
                "frame_id": str(frame_id),
                "frame_number": int(hit.get("frame_number", 0)),
                "timestamp_ms": int(hit.get("timestamp_ms", 0)),
                "confidence": max(0.0, min(1.0, confidence)),
                "frame_image_url": str(hit.get("frame_image_url", hit.get("image_url", ""))),
                "fps": float(hit.get("fps", 25.0)),
            }
            output.append(_strip_private(row))
        return output

    async def _genre_expr(self, video_genre: str) -> str | None:
        if not video_genre or video_genre == "All":
            return None
        video_ids = await postgres_client.fetch_video_ids_by_genre(video_genre)
        if not video_ids:
            return 'video_id == "__none__"'
        quoted = ", ".join(f'"{video_id}"' for video_id in video_ids)
        return f"video_id in [{quoted}]"

    @staticmethod
    async def _hydrate_frames(hits: list[dict]) -> None:
        rows = await postgres_client.fetch_video_metadata(list({hit["video_id"] for hit in hits}))
        videos = {row["video_id"]: row for row in rows}
        for hit in hits:
            video = videos.get(hit["video_id"], {})
            # Only overwrite when PostgreSQL actually knows better. The zip
            # ingest denormalizes youtube_id/fps straight into each Milvus
            # record precisely so playback works without a videos row — and
            # with --skip-postgres there is no row at all, so an unconditional
            # assignment blanked a value that was already correct.
            if video.get("youtube_id"):
                hit["youtube_id"] = str(video["youtube_id"])
            else:
                hit["youtube_id"] = str(hit.get("youtube_id", ""))
            hit["fps"] = float(video.get("fps") or hit.get("fps") or 25.0)
            # Lots ingested from *_results.zip carry no JPG and no image_url
            # (ingest_zip_pipeline_results.py leaves it blank on purpose, so the
            # URL can't go stale when the serving mechanism changes). Derive it
            # here from what every hit already has. Same rule as local-backend.
            if not hit.get("image_url"):
                hit["image_url"] = f"/api/zip-frame/{hit['video_id']}/{hit['timestamp_ms']}"

    async def get_raw_data(self, query_groups: list[dict], limit: int = 1000, video_genre: str = "All") -> dict:
        """
        Fetch multi-modal raw data for all query groups from local databases.
        Returns a unified dict that strategies receive as ``raw_data``.

        Args:
            video_genre: optional genre filter (e.g. "Ẩm thực", "Công nghệ").
                         "All" or "" disables filtering.
        """
        vector_algorithm = _requested_vector_algorithm(query_groups)
        all_frame_ids: set[str] = set()
        all_video_ids: set[str] = set()
        frames: list[dict] = []
        ocr: list[dict] = []
        transcripts: list[dict] = []
        transcript_chunks: list[dict] = []

        # Build combined text query for transcript chunk search
        combined_text_query = " ".join(
            g.get("text_query", "").strip() for g in query_groups
        ).strip()

        # Resolve genre → video_ids for Milvus scalar filter
        genre_expr: str | None = None
        if video_genre and video_genre != "All":
            genre_video_ids = await postgres_client.fetch_video_ids_by_genre(video_genre)
            if genre_video_ids:
                quoted = ", ".join(f'"{vid}"' for vid in genre_video_ids)
                genre_expr = f"video_id in [{quoted}]"
                logger.info("Genre filter: %s → %d videos", video_genre, len(genre_video_ids))
            else:
                logger.info("Genre filter: %s → 0 videos, skipping", video_genre)

        for group_index, group in enumerate(query_groups):
            semantic_query = group.get("semantic_query", "").strip()
            text_query = group.get("text_query", "").strip()

            if semantic_query:
                query_vector = await self._encode_text(semantic_query)
                search_limit = max(int(limit), 1)
                timer_start = time.monotonic()
                if vector_algorithm in {"hnsw", "flat", "scann"}:
                    hits = await self._run_db(milvus_client.vector_search,
                        self._get_collection(vector_algorithm),
                        query_vector.tolist(),
                        top_k=search_limit,
                        algorithm=vector_algorithm,
                        expr=genre_expr,
                    )
                else:
                    hits = self._get_cagra().search(query_vector, top_k=search_limit)
                logger.info(
                    "[TIMER] vector_search %.3f ms backend=%s group=%s top_k=%s hits=%s",
                    (time.monotonic() - timer_start) * 1000,
                    vector_algorithm,
                    group_index,
                    search_limit,
                    len(hits),
                )
                for hit in hits:
                    hit["_query_group_index"] = group_index
                frames.extend(hits)
                all_frame_ids.update(h["frame_id"] for h in hits)
                all_video_ids.update(h["video_id"] for h in hits)
                logger.info(
                    "%s semantic query group=%s returned %s hits",
                    vector_algorithm,
                    group_index,
                    len(hits),
                )

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

        # Transcript chunk vector search (topic-based, one search across all text queries)
        if self._transcript_search is not None and combined_text_query:
            try:
                chunk_results = await self._transcript_search.search(
                    combined_text_query,
                    top_k=limit,
                )
                transcript_chunks = chunk_results
                all_video_ids.update(h["video_id"] for h in chunk_results)
                frames.extend(self._frames_for_transcript_chunks(chunk_results, limit=limit))
                logger.info(
                    "Transcript chunk search returned %s chunks",
                    len(chunk_results),
                )
            except Exception as exc:
                logger.exception("Transcript chunk search failed: %s", exc)

        # Fetch video metadata for all referenced videos
        video_rows = await postgres_client.fetch_video_metadata(list(all_video_ids))
        videos = {v["video_id"]: dict(v) for v in video_rows}
        for frame in frames:
            if frame.pop("_timestamp_from_fps", False):
                fps = float(videos.get(frame["video_id"], {}).get("fps") or 25.0)
                frame["timestamp_ms"] = int(frame["frame_number"] / fps * 1000)

        return {
            "frames":             _dedupe_frames(frames),
            "ocr":                _dedupe_by_key(ocr, "frame_id"),
            "transcripts":        transcripts,
            "transcript_chunks":  transcript_chunks,
            "videos":             videos,
            "video_genre":        video_genre or "All",
        }

    def _frames_for_transcript_chunks(self, chunks: list[dict], limit: int) -> list[dict]:
        frames = []
        if not chunks:
            return frames
        per_chunk_limit = max(1, min(20, limit // max(1, len(chunks))))
        for chunk in chunks:
            chunk_frames = milvus_client.query_frames_in_time_range(
                self._collection,
                chunk["video_id"],
                int(chunk["start_time_ms"]),
                int(chunk["end_time_ms"]),
                limit=per_chunk_limit,
            )
            frames.extend(chunk_frames)
            if len(frames) >= limit:
                break
        return frames[:limit]

    def _frames_for_transcripts(self, transcript_hits: list[dict], limit: int) -> list[dict]:
        frames = []
        if not transcript_hits:
            return frames

        per_interval_limit = max(1, min(20, limit // max(1, len(transcript_hits))))
        for hit in transcript_hits:
            interval_frames = milvus_client.query_frames_in_time_range(
                self._metadata_collection(),
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


def _requested_vector_algorithm(query_groups: list[dict]) -> str:
    for group in query_groups:
        algorithm = str(
            group.get("_vector_search_algorithm")
            or ""
        ).strip().lower()
        if algorithm:
            if algorithm not in VALID_VECTOR_SEARCH_ALGORITHMS:
                raise ValueError("vector_search_algorithm must be 'hnsw', 'flat', 'cagra', or 'scann'")
            return algorithm
    return DEFAULT_VECTOR_SEARCH_ALGORITHM


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


def _strip_private(value):
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value
