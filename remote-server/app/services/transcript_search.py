from __future__ import annotations

import logging
import os
import threading
import time
from functools import lru_cache

import numpy as np

from app.db import milvus_client, postgres_client

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("TRANSCRIPT_MODEL_ID", "intfloat/multilingual-e5-small")
MODEL_DEVICE = os.getenv("TRANSCRIPT_MODEL_DEVICE", "cpu")
EXPECTED_DIM = int(os.getenv("TRANSCRIPT_VECTOR_DIM", "384"))

TOPICS = milvus_client.TOPICS


class TranscriptSearchService:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._topic_vectors: np.ndarray | None = None
        self._collection = None

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

    def encode_passage(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(
            f"passage: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

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

        query_vector = self.encode_query(query)

        if topic_filter is None:
            topic_filter = self.classify_topic(query_vector)

        chunk_hits = milvus_client.search_transcript_chunks(
            self._collection,
            query_vector.tolist(),
            top_k=top_k,
            topic_filter=topic_filter,
        )
        logger.info(
            "[TIMER] transcript_chunk_search %.3f ms topic=%s top_k=%s hits=%s",
            (time.monotonic() - timer_start) * 1000,
            topic_filter,
            top_k,
            len(chunk_hits),
        )

        if not chunk_hits:
            return []

        chunk_ids = [int(h["chunk_id"]) for h in chunk_hits]
        metadata_rows = await postgres_client.fetch_transcript_chunks_by_ids(chunk_ids)
        metadata_by_id = {row["chunk_id"]: row for row in metadata_rows}

        results = []
        for hit in chunk_hits:
            meta = metadata_by_id.get(hit["chunk_id"], {})
            results.append({
                "chunk_id":         hit["chunk_id"],
                "video_id":         hit["video_id"],
                "topic":            hit["topic"],
                "start_time_ms":    hit["start_time_ms"],
                "end_time_ms":      hit["end_time_ms"],
                "text":             meta.get("raw_text", ""),
                "score":            round(float(hit["score"]), 4),
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
