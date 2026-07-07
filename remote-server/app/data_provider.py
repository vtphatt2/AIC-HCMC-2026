import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.db import milvus_client, postgres_client
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
                keyframe_dir = Path(os.getenv("FRAME_STATIC_DIR", ""))
                if not keyframe_dir.is_dir():
                    sample_root = Path(__file__).parent.parent.parent / "AIC2026_sample"
                    keyframe_dir = sample_root / "keyframes" / "keyframes"
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

    def _get_cagra(self):
        if self._cagra is not None:
            return self._cagra
        with self._cagra_lock:
            if self._cagra is None:
                from app.db.cagra_client import CagraClient

                self._cagra = CagraClient()
        return self._cagra

    def _get_collection(self, algorithm: str):
        algorithm = milvus_client.normalize_algorithm(algorithm)
        if algorithm not in self._collections:
            collection_name = milvus_client.collection_name_for_algorithm(algorithm)
            if not milvus_client.has_collection_for_algorithm(algorithm):
                raise ValueError(
                    f"{algorithm.upper()} collection '{collection_name}' is not available. "
                    "Build it with scripts/ingest_embeddings_to_milvus.py --vector-index all."
                )
            self._collections[algorithm] = milvus_client.get_collection_for_name(collection_name)
        return self._collections[algorithm]

    def _metadata_collection(self):
        algorithm = DEFAULT_VECTOR_SEARCH_ALGORITHM
        if algorithm == "cagra":
            algorithm = "hnsw"
        return self._get_collection(algorithm)

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
                    hits = milvus_client.vector_search(
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

    async def get_raw_data_two_stage(
        self,
        query_groups: list[dict],
        limit: int = 1000,
        video_genre: str = "All",
        stage1_top_k: int = 20,
    ) -> dict:
        """
        Two-stage late fusion retrieval pipeline.

        Stage 1 (Coarse): Transcript vector search → candidate video_ids + time windows
        Stage 2 (Fine): PE-Core visual search filtered to candidate videos only

        Returns raw_data dict with additional keys:
          - "stage1_chunks": Stage 1 transcript chunk results
          - "stage1_video_ids": set of candidate video_ids from Stage 1
          - "stage2_frames": Stage 2 frame results (filtered to candidate videos)
        """
        vector_algorithm = _requested_vector_algorithm(query_groups)
        frames: list[dict] = []
        ocr: list[dict] = []
        transcripts: list[dict] = []
        transcript_chunks: list[dict] = []
        stage1_chunks: list[dict] = []
        stage1_video_ids: set[str] = set()
        stage2_frames: list[dict] = []
        all_video_ids: set[str] = set()

        combined_text_query = " ".join(
            g.get("text_query", "").strip() for g in query_groups
        ).strip()
        combined_semantic_query = " ".join(
            g.get("semantic_query", "").strip() for g in query_groups
        ).strip()

        genre_expr: str | None = None
        if video_genre and video_genre != "All":
            genre_video_ids = await postgres_client.fetch_video_ids_by_genre(video_genre)
            if genre_video_ids:
                quoted = ", ".join(f'"{vid}"' for vid in genre_video_ids)
                genre_expr = f"video_id in [{quoted}]"
                logger.info("Genre filter: %s → %d videos", video_genre, len(genre_video_ids))

        # ── Stage 1: Transcript chunk search → candidate videos ──
        if self._transcript_search is not None and combined_text_query:
            try:
                chunks, candidate_vids = await self._transcript_search.search_for_candidates(
                    combined_text_query,
                    top_k=stage1_top_k,
                )
                stage1_chunks = chunks
                stage1_video_ids = candidate_vids
                logger.info(
                    "Stage 1: transcript search returned %d chunks, %d candidate videos",
                    len(chunks), len(candidate_vids),
                )
            except Exception as exc:
                logger.exception("Stage 1 transcript search failed: %s", exc)

        # ── Build Milvus expr for Stage 2 ──
        stage2_expr_parts: list[str] = []
        if stage1_video_ids:
            quoted_vids = ", ".join(f'"{vid}"' for vid in stage1_video_ids)
            stage2_expr_parts.append(f"video_id in [{quoted_vids}]")
        if genre_expr:
            stage2_expr_parts.append(genre_expr)

        stage2_expr = " and ".join(stage2_expr_parts) if stage2_expr_parts else None

        # ── Stage 2: PE-Core visual search filtered to candidate videos ──
        for group_index, group in enumerate(query_groups):
            semantic_query = group.get("semantic_query", "").strip()

            if semantic_query:
                query_vector = await self._encode_text(semantic_query)
                search_limit = max(int(limit), 1)
                timer_start = time.monotonic()

                if vector_algorithm in {"hnsw", "flat", "scann"}:
                    hits = milvus_client.vector_search(
                        self._get_collection(vector_algorithm),
                        query_vector.tolist(),
                        top_k=search_limit,
                        algorithm=vector_algorithm,
                        expr=stage2_expr,
                    )
                else:
                    hits = self._get_cagra().search(query_vector, top_k=search_limit)
                    if stage1_video_ids:
                        hits = [h for h in hits if h["video_id"] in stage1_video_ids]

                logger.info(
                    "[TIMER] stage2_vector_search %.3f ms backend=%s group=%s top_k=%s hits=%s expr=%s",
                    (time.monotonic() - timer_start) * 1000,
                    vector_algorithm,
                    group_index,
                    search_limit,
                    len(hits),
                    stage2_expr is not None,
                )
                for hit in hits:
                    hit["_query_group_index"] = group_index
                    hit["_stage"] = 2
                stage2_frames.extend(hits)
                frames.extend(hits)
                all_video_ids.update(h["video_id"] for h in hits)

        # ── Also fetch global visual results for fallback ──
        if not stage2_frames and combined_semantic_query:
            logger.info("Stage 2 returned 0 results, falling back to global visual search")
            for group_index, group in enumerate(query_groups):
                semantic_query = group.get("semantic_query", "").strip()
                if semantic_query:
                    query_vector = await self._encode_text(semantic_query)
                    search_limit = max(int(limit), 1)
                    if vector_algorithm in {"hnsw", "flat", "scann"}:
                        hits = milvus_client.vector_search(
                            self._get_collection(vector_algorithm),
                            query_vector.tolist(),
                            top_k=search_limit,
                            algorithm=vector_algorithm,
                            expr=genre_expr,
                        )
                    else:
                        hits = self._get_cagra().search(query_vector, top_k=search_limit)
                    for hit in hits:
                        hit["_query_group_index"] = group_index
                        hit["_stage"] = "fallback"
                    frames.extend(hits)
                    all_video_ids.update(h["video_id"] for h in hits)

        # ── Fetch transcript chunk metadata ──
        if stage1_chunks:
            chunk_ids = [int(h["chunk_id"]) for h in stage1_chunks]
            metadata_rows = await postgres_client.fetch_transcript_chunks_by_ids(chunk_ids)
            metadata_by_id = {row["chunk_id"]: row for row in metadata_rows}
            for chunk in stage1_chunks:
                meta = metadata_by_id.get(chunk["chunk_id"], {})
                chunk["text"] = meta.get("raw_text", "")
            transcript_chunks = stage1_chunks
            all_video_ids.update(h["video_id"] for h in stage1_chunks)

        # ── Fetch text query results (OCR + transcript text) ──
        text_query = " ".join(g.get("text_query", "").strip() for g in query_groups).strip()
        if text_query:
            ocr_hits = await postgres_client.search_ocr_text(text_query, limit=limit)
            ocr.extend(ocr_hits)
            all_video_ids.update(h["video_id"] for h in ocr_hits)

            transcript_hits = await postgres_client.search_transcript_text(text_query, limit=limit)
            transcripts.extend(transcript_hits)
            all_video_ids.update(h["video_id"] for h in transcript_hits)

        # ── Fetch video metadata ──
        video_rows = await postgres_client.fetch_video_metadata(list(all_video_ids))
        videos = {v["video_id"]: dict(v) for v in video_rows}

        return {
            "frames":             _dedupe_frames(frames),
            "ocr":                _dedupe_by_key(ocr, "frame_id"),
            "transcripts":        transcripts,
            "transcript_chunks":  transcript_chunks,
            "videos":             videos,
            "video_genre":        video_genre or "All",
            "stage1_chunks":      stage1_chunks,
            "stage1_video_ids":   stage1_video_ids,
            "stage2_frames":      stage2_frames,
        }

    def _frames_for_transcript_chunks(self, chunks: list[dict], limit: int) -> list[dict]:
        frames = []
        if not chunks:
            return frames
        per_chunk_limit = max(1, min(20, limit // max(1, len(chunks))))
        for chunk in chunks:
            chunk_frames = milvus_client.query_frames_in_time_range(
                self._metadata_collection(),
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
