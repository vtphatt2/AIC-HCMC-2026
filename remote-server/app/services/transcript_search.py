from __future__ import annotations

import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.db import milvus_client, postgres_client

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("TRANSCRIPT_MODEL_ID", "intfloat/multilingual-e5-small")
MODEL_DEVICE = os.getenv("TRANSCRIPT_MODEL_DEVICE", "cpu")
EXPECTED_DIM = int(os.getenv("TRANSCRIPT_VECTOR_DIM", "384"))
NEAREST_FRAME_WINDOW_MS = int(os.getenv("TRANSCRIPT_NEAREST_FRAME_WINDOW_MS", "3000"))
NEAREST_FRAME_QUERY_LIMIT = int(os.getenv("TRANSCRIPT_NEAREST_FRAME_QUERY_LIMIT", "256"))

TOPICS = milvus_client.TOPICS


class TranscriptSearchService:
    def __init__(self, keyframe_dir: Path | None = None):
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._topic_vectors: np.ndarray | None = None
        self._collection = None
        self._frames_collection = None
        self._keyframe_dir = keyframe_dir

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            timer_start = time.monotonic()
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for transcript search. "
                    "Install: pip install sentence-transformers>=2.2.0"
                )

            self._model = SentenceTransformer(MODEL_ID, device=MODEL_DEVICE)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != EXPECTED_DIM:
                raise RuntimeError(
                    f"Model {MODEL_ID} has dim {actual_dim}, expected {EXPECTED_DIM}"
                )

            topic_embeddings = self._model.encode(
                TOPICS,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._topic_vectors = np.asarray(topic_embeddings, dtype="float32")

            self._loaded = True
            logger.info(
                "[TIMER] transcript_model_load %.3f ms model=%s device=%s dim=%s topics=%s",
                (time.monotonic() - timer_start) * 1000,
                MODEL_ID,
                MODEL_DEVICE,
                actual_dim,
                len(TOPICS),
            )

    def _ensure_collection(self) -> None:
        if self._collection is None:
            self._collection = milvus_client.get_transcript_collection()

    def _ensure_frames_collection(self) -> None:
        if self._frames_collection is None:
            try:
                self._frames_collection = milvus_client.get_collection()
            except Exception:
                logger.warning("video_frames collection not available for transcript frame lookup")

    def encode_passage(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(
            f"passage: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    @lru_cache(maxsize=128)
    def _cached_encode_query(self, text: str) -> bytes:
        vector = self._encode_query_uncached(text)
        return vector.tobytes()

    def _encode_query_uncached(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        buffer = self._cached_encode_query(text)
        return np.frombuffer(buffer, dtype="float32")

    def classify_topic(self, vector: np.ndarray) -> str:
        self._ensure_loaded()
        similarities = self._topic_vectors @ vector
        best_idx = int(similarities.argmax())
        return TOPICS[best_idx]

    def classify_topic_text(self, text: str) -> str:
        vector = self.encode_passage(text)
        return self.classify_topic(vector)

    async def search(
        self,
        query: str,
        top_k: int = 100,
        topic_filter: str | None = None,
    ) -> list[dict]:
        timer_start = time.monotonic()
        self._ensure_loaded()
        self._ensure_collection()
        self._ensure_frames_collection()

        query_vector = self.encode_query(query)

        chunk_hits = milvus_client.search_transcript_chunks(
            self._collection,
            query_vector.tolist(),
            top_k=top_k,
            topic_filter=topic_filter,
        )
        logger.info(
            "[TIMER] transcript_chunk_search %.3f ms topic=%s top_k=%s hits=%s",
            (time.monotonic() - timer_start) * 1000,
            topic_filter or "All",
            top_k,
            len(chunk_hits),
        )

        if not chunk_hits:
            return []

        chunk_ids = [int(h["chunk_id"]) for h in chunk_hits]
        metadata_rows = await postgres_client.fetch_transcript_chunks_by_ids(chunk_ids)
        metadata_by_id = {row["chunk_id"]: row for row in metadata_rows}

        # Fetch video metadata for youtube_id lookup
        all_video_ids = list({h["video_id"] for h in chunk_hits})
        video_rows = await postgres_client.fetch_video_metadata(all_video_ids)
        video_meta = {v["video_id"]: v for v in video_rows}

        results = []
        for hit in chunk_hits:
            meta = metadata_by_id.get(hit["chunk_id"], {})
            vmeta = video_meta.get(hit["video_id"], {})
            if not vmeta:
                logger.warning("video_id=%s not found in videos table — youtube_id will be empty", hit["video_id"])
            mid_ms = (int(hit["start_time_ms"]) + int(hit["end_time_ms"])) // 2
            frame_info = self._nearest_frame(hit["video_id"], mid_ms)
            results.append({
                "chunk_id":          hit["chunk_id"],
                "video_id":          hit["video_id"],
                "youtube_id":        str(vmeta.get("youtube_id", "")),
                "topic":             hit["topic"],
                "start_time_ms":     hit["start_time_ms"],
                "end_time_ms":       hit["end_time_ms"],
                "text":              meta.get("raw_text", ""),
                "score":             round(float(hit["score"]), 4),
                "frame_image_url":   frame_info.get("image_url", ""),
                "frame_number":      frame_info.get("frame_number", 0),
                "nearest_timestamp_ms": frame_info.get("timestamp_ms"),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def encode_batch_passages(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        prefixed = [f"passage: {t}" for t in texts]
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).astype("float32")

    @lru_cache(maxsize=1024)
    def _nearest_frame(self, video_id: str, target_ms: int) -> dict:
        if self._frames_collection is not None:
            try:
                frames = milvus_client.query_frames_in_time_range(
                    self._frames_collection,
                    video_id,
                    max(0, target_ms - NEAREST_FRAME_WINDOW_MS),
                    target_ms + NEAREST_FRAME_WINDOW_MS,
                    limit=NEAREST_FRAME_QUERY_LIMIT,
                )
                if frames:
                    nearest = min(frames, key=lambda f: abs(int(f["timestamp_ms"]) - target_ms))
                    return {
                        "image_url":    nearest.get("image_url", ""),
                        "frame_number": nearest.get("frame_number", 0),
                        "timestamp_ms": nearest.get("timestamp_ms"),
                    }
            except Exception:
                logger.warning("Milvus frame query failed for video_id=%s, using filesystem fallback", video_id)
        return self._nearest_frame_fallback(video_id, target_ms)

    def _nearest_frame_fallback(self, video_id: str, target_ms: int) -> dict:
        if self._keyframe_dir is None:
            return {}
        frame_dir = self._keyframe_dir / video_id
        if not frame_dir.is_dir():
            return {}
        try:
            entries = list(frame_dir.glob("*.jpg"))
            if not entries:
                return {}
            target_frame = int(target_ms / 1000 * 25)
            best_path = None
            best_diff = float("inf")
            for p in entries:
                try:
                    fnum = int(p.stem)
                except ValueError:
                    continue
                diff = abs(fnum - target_frame)
                if diff < best_diff:
                    best_diff = diff
                    best_path = p
            if best_path is None:
                return {}
            frame_number = int(best_path.stem)
            timestamp_ms = int(frame_number / 25 * 1000)
            image_url = f"/static/frames/{video_id}/{best_path.name}"
            return {
                "image_url":    image_url,
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
            }
        except Exception:
            logger.warning("Filesystem frame fallback failed for video_id=%s", video_id)
            return {}

    def warmup(self, query: str = "warmup query", passes: int = 5) -> None:
        load_start = time.monotonic()
        was_loaded = self._loaded
        self._ensure_loaded()
        model_load_ms = 0.0 if was_loaded else (time.monotonic() - load_start) * 1000

        encode_start = time.monotonic()
        passes = max(1, int(passes))
        for _ in range(passes):
            # Bypass the cache for warmup passes to ensure GPU execution is fully evaluated
            self._encode_query_uncached(query)
        encode_ms = (time.monotonic() - encode_start) * 1000

        logger.info(
            "[TIMER] warmup_transcript_search model_load_ms=%.3f encode_ms=%.3f passes=%s device=%s",
            model_load_ms,
            encode_ms,
            passes,
            MODEL_DEVICE,
        )
