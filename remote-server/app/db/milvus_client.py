from __future__ import annotations

import logging
import os
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "video_frames")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1280"))  # PE-Core-bigG-14-448 produces 1280-dim
METRIC_TYPE = "COSINE"
INDEX_TYPE = "HNSW"
INDEX_PARAMS = {"M": 16, "efConstruction": 256}
SEARCH_PARAMS = {"ef": 256}
DEEP_SEARCH_TOP_K = 50
DEEP_SEARCH_EF = 512


def connect() -> None:
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
    )
    logger.info("Connected to Milvus collection=%s dim=%s", COLLECTION_NAME, VECTOR_DIM)


def get_collection() -> Collection:
    col = Collection(COLLECTION_NAME)
    col.load()
    return col


def create_collection_if_missing() -> Collection:
    """Create the video_frames collection with an HNSW index if it does not exist."""
    if utility.has_collection(COLLECTION_NAME):
        logger.info("Milvus collection already exists: %s", COLLECTION_NAME)
        return get_collection()

    fields = [
        FieldSchema(name="frame_id",     dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="video_id",     dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="video_genre",  dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="frame_number", dtype=DataType.INT64),
        FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
        FieldSchema(name="image_url",    dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(fields, description="Frame-level visual embeddings (PE-Core-bigG-14-448)")
    col = Collection(COLLECTION_NAME, schema)

    col.create_index(
        "vector",
        {
            "metric_type": METRIC_TYPE,
            "index_type": INDEX_TYPE,
            "params": INDEX_PARAMS,
        },
    )
    try:
        col.create_index("video_genre", {"index_type": "INVERTED"})
    except Exception:
        logger.warning("Could not create INVERTED index on video_genre field; scalar filtering may be slow.")

    col.load()
    logger.info(
        "Created Milvus collection=%s dim=%s index=%s metric=%s params=%s",
        COLLECTION_NAME,
        VECTOR_DIM,
        INDEX_TYPE,
        METRIC_TYPE,
        INDEX_PARAMS,
    )
    return col


def drop_collection_if_exists() -> bool:
    if not utility.has_collection(COLLECTION_NAME):
        return False
    utility.drop_collection(COLLECTION_NAME)
    logger.warning("Dropped Milvus collection: %s", COLLECTION_NAME)
    return True


def upsert_frame_vectors(collection: Collection, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    collection.upsert(
        [
            [str(r["frame_id"]) for r in records],
            [str(r["video_id"]) for r in records],
            [str(r.get("video_genre", "")) for r in records],
            [int(r["frame_number"]) for r in records],
            [int(r["timestamp_ms"]) for r in records],
            [str(r.get("image_url", "")) for r in records],
            [r["vector"] for r in records],
        ]
    )


def vector_search(
    collection: Collection,
    query_vector: list[float],
    top_k: int = 100,
    expr: str | None = None,
) -> list[dict]:
    top_k = max(1, int(top_k))
    base_ef = DEEP_SEARCH_EF if top_k >= DEEP_SEARCH_TOP_K else int(SEARCH_PARAMS["ef"])
    search_params = {"ef": max(base_ef, top_k)}

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": METRIC_TYPE, "params": search_params},
        limit=top_k,
        expr=expr,
        output_fields=["frame_id", "video_id", "frame_number", "timestamp_ms", "image_url"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "frame_id":     hit.entity.get("frame_id"),
            "video_id":     hit.entity.get("video_id"),
            "frame_number": hit.entity.get("frame_number"),
            "timestamp_ms": hit.entity.get("timestamp_ms"),
            "image_url":    hit.entity.get("image_url"),
            "score":        hit.score,
        })
    return hits


def query_frames_in_time_range(
    collection: Collection,
    video_id: str,
    start_ms: int,
    end_ms: int,
    limit: int = 100,
) -> list[dict]:
    expr = (
        f'video_id == "{video_id}" '
        f"and timestamp_ms >= {int(start_ms)} "
        f"and timestamp_ms <= {int(end_ms)}"
    )
    rows = collection.query(
        expr=expr,
        output_fields=["frame_id", "video_id", "frame_number", "timestamp_ms", "image_url"],
        limit=max(1, int(limit)),
    )
    rows.sort(key=lambda row: row["timestamp_ms"])
    return [dict(row) for row in rows]


# ── Transcript Chunks collection ──────────────────────────────────────────────

TRANSCRIPT_CHUNKS_COLLECTION = os.getenv("MILVUS_TRANSCRIPT_COLLECTION", "transcript_chunks")
TRANSCRIPT_VECTOR_DIM = int(os.getenv("TRANSCRIPT_VECTOR_DIM", "384"))
TRANSCRIPT_METRIC_TYPE = "COSINE"
TRANSCRIPT_INDEX_TYPE = "HNSW"
TRANSCRIPT_INDEX_PARAMS = {"M": 16, "efConstruction": 256}
TRANSCRIPT_SEARCH_PARAMS = {"ef": 256}

TOPICS = [
    "Ẩm thực", "Công nghệ", "Du lịch", "Thể thao", "Giáo dục",
    "Kinh tế", "Sức khỏe", "Giải trí", "Thời sự", "Văn hóa",
    "Đời sống", "Môi trường", "Giao thông", "Pháp luật",
]


def get_transcript_collection() -> Collection:
    col = Collection(TRANSCRIPT_CHUNKS_COLLECTION)
    col.load()
    return col


def create_transcript_collection_if_missing() -> Collection:
    if utility.has_collection(TRANSCRIPT_CHUNKS_COLLECTION):
        logger.info("Milvus collection already exists: %s", TRANSCRIPT_CHUNKS_COLLECTION)
        return get_transcript_collection()

    fields = [
        FieldSchema(name="chunk_id",      dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="video_id",      dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="topic",         dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="start_time_ms", dtype=DataType.INT64),
        FieldSchema(name="end_time_ms",   dtype=DataType.INT64),
        FieldSchema(name="vector",        dtype=DataType.FLOAT_VECTOR, dim=TRANSCRIPT_VECTOR_DIM),
    ]
    schema = CollectionSchema(fields, description="Transcript chunks with topic-labelled embeddings")
    col = Collection(TRANSCRIPT_CHUNKS_COLLECTION, schema)

    col.create_index(
        "vector",
        {
            "metric_type": TRANSCRIPT_METRIC_TYPE,
            "index_type": TRANSCRIPT_INDEX_TYPE,
            "params": TRANSCRIPT_INDEX_PARAMS,
        },
    )
    try:
        col.create_index("topic", {"index_type": "INVERTED"})
    except Exception:
        logger.warning("Could not create INVERTED index on topic field; scalar filtering may be slow.")

    col.load()
    logger.info(
        "Created Milvus collection=%s dim=%s index=%s metric=%s",
        TRANSCRIPT_CHUNKS_COLLECTION,
        TRANSCRIPT_VECTOR_DIM,
        TRANSCRIPT_INDEX_TYPE,
        TRANSCRIPT_METRIC_TYPE,
    )
    return col


def drop_transcript_collection_if_exists() -> bool:
    if not utility.has_collection(TRANSCRIPT_CHUNKS_COLLECTION):
        return False
    utility.drop_collection(TRANSCRIPT_CHUNKS_COLLECTION)
    logger.warning("Dropped Milvus collection: %s", TRANSCRIPT_CHUNKS_COLLECTION)
    return True


def upsert_transcript_chunks(collection: Collection, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    collection.upsert(
        [
            [int(r["chunk_id"]) for r in records],
            [str(r["video_id"]) for r in records],
            [str(r["topic"]) for r in records],
            [int(r["start_time_ms"]) for r in records],
            [int(r["end_time_ms"]) for r in records],
            [r["vector"] for r in records],
        ]
    )


def search_transcript_chunks(
    collection: Collection,
    query_vector: list[float],
    top_k: int = 100,
    topic_filter: str | None = None,
) -> list[dict]:
    top_k = max(1, int(top_k))
    search_params = {"ef": max(int(TRANSCRIPT_SEARCH_PARAMS["ef"]), top_k)}
    expr = f'topic == "{topic_filter}"' if topic_filter else None

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": TRANSCRIPT_METRIC_TYPE, "params": search_params},
        limit=top_k,
        expr=expr,
        output_fields=["chunk_id", "video_id", "topic", "start_time_ms", "end_time_ms"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id":      hit.entity.get("chunk_id"),
            "video_id":      hit.entity.get("video_id"),
            "topic":         hit.entity.get("topic"),
            "start_time_ms": hit.entity.get("start_time_ms"),
            "end_time_ms":   hit.entity.get("end_time_ms"),
            "score":         hit.score,
        })
    return hits
