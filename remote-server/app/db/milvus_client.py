from __future__ import annotations

import json
import logging
import os
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.db.milvus_compat import query_frame_vector_rows

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "video_frames")
SUBTITLED_COLLECTION_NAME = os.getenv("MILVUS_SUBTITLED_COLLECTION", "video_frames_subtitled")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1280"))  # PE-Core-bigG-14-448 produces 1280-dim
METRIC_TYPE = "COSINE"
DEFAULT_ALGORITHM = {
    "milvus": "hnsw",
    "hnsw": "hnsw",
    "flat": "flat",
    "scann": "scann",
}.get(os.getenv("VECTOR_SEARCH_BACKEND", "milvus").lower(), "hnsw")
VECTOR_INDEXES = {
    "hnsw": {
        "collection": os.getenv("MILVUS_COLLECTION_HNSW") or COLLECTION_NAME,
        "index_type": "HNSW",
        "index_params": {"M": 16, "efConstruction": 256},
    },
    "flat": {
        "collection": os.getenv("MILVUS_COLLECTION_FLAT") or f"{COLLECTION_NAME}_flat",
        "index_type": "FLAT",
        "index_params": {},
    },
    "scann": {
        "collection": os.getenv("MILVUS_COLLECTION_SCANN") or f"{COLLECTION_NAME}_scann",
        "index_type": "SCANN",
        "index_params": {"nlist": 127, "with_raw_data": True},
    },
}
INDEX_TYPE = VECTOR_INDEXES[DEFAULT_ALGORITHM]["index_type"]
INDEX_PARAMS = VECTOR_INDEXES[DEFAULT_ALGORITHM]["index_params"]
SEARCH_PARAMS = {"ef": int(os.getenv("MILVUS_HNSW_EF", "256"))}
DEEP_SEARCH_TOP_K = 50
DEEP_SEARCH_EF = int(os.getenv("MILVUS_HNSW_DEEP_EF", "512"))
if not all(1 <= ef <= 32768 for ef in (SEARCH_PARAMS["ef"], DEEP_SEARCH_EF)):
    raise ValueError("Milvus HNSW ef values must be between 1 and 32768")


def connect() -> None:
    lite_path = os.getenv("MILVUS_LITE_PATH", "").strip()
    if lite_path:
        connections.connect(uri=lite_path)
        logger.info("Connected to Milvus Lite at %s (dim=%s)", lite_path, VECTOR_DIM)
        return
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
    )
    logger.info("Connected to Milvus collection=%s dim=%s", COLLECTION_NAME, VECTOR_DIM)


def get_collection(channel: str = "raw.semantic") -> Collection:
    col = Collection(collection_name_for_algorithm(channel=channel))
    col.load()
    return col


def get_collection_for_name(collection_name: str) -> Collection:
    col = Collection(collection_name)
    col.load()
    return col


def collection_name_for_algorithm(
    algorithm: str | None = None,
    channel: str = "raw.semantic",
) -> str:
    algorithm = normalize_algorithm(algorithm or DEFAULT_ALGORITHM)
    if channel == "raw.semantic":
        return VECTOR_INDEXES[algorithm]["collection"]
    if channel != "subtitled.semantic":
        raise ValueError(f"Unknown visual channel '{channel}'")
    env_name = f"MILVUS_SUBTITLED_COLLECTION_{algorithm.upper()}"
    suffix = "" if algorithm == "hnsw" else f"_{algorithm}"
    return os.getenv(env_name) or f"{SUBTITLED_COLLECTION_NAME}{suffix}"


def index_config(
    algorithm: str | None = None,
    channel: str = "raw.semantic",
) -> dict[str, Any]:
    algorithm = normalize_algorithm(algorithm or DEFAULT_ALGORITHM)
    return {
        **VECTOR_INDEXES[algorithm],
        "collection": collection_name_for_algorithm(algorithm, channel),
    }


def normalize_algorithm(algorithm: str) -> str:
    algorithm = str(algorithm).strip().lower()
    if algorithm == "milvus":
        algorithm = "hnsw"
    if algorithm not in VECTOR_INDEXES:
        raise ValueError(f"Milvus vector algorithm must be one of {list(VECTOR_INDEXES.keys())}")
    return algorithm


def available_milvus_algorithms() -> dict[str, bool]:
    return {
        algorithm: has_collection_for_algorithm(algorithm)
        for algorithm in VECTOR_INDEXES
    }


def has_collection_for_algorithm(algorithm: str, channel: str = "raw.semantic") -> bool:
    return utility.has_collection(collection_name_for_algorithm(algorithm, channel))


def create_collection_if_missing(
    algorithm: str | None = None,
    channel: str = "raw.semantic",
) -> Collection:
    """Create the configured Milvus collection/index if it does not exist."""
    config = index_config(algorithm, channel)
    collection_name = config["collection"]
    if utility.has_collection(collection_name):
        logger.info("Milvus collection already exists: %s", collection_name)
        col = Collection(collection_name)
        col.load()
        return col

    fields = [
        FieldSchema(name="frame_id",     dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="video_id",     dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="video_genre",  dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="frame_number", dtype=DataType.INT64),
        FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
        FieldSchema(name="image_url",    dtype=DataType.VARCHAR, max_length=256),
        # Denormalized from Postgres on purpose: local-backend's SAMPLE mode
        # queries Milvus only (no Postgres), so YouTube-primary playback
        # needs youtube_id available straight from a search hit.
        FieldSchema(name="youtube_id",   dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(fields, description="Frame-level visual embeddings (PE-Core-bigG-14-448)")
    col = Collection(collection_name, schema)

    col.create_index(
        "vector",
        {
            "metric_type": METRIC_TYPE,
            "index_type": config["index_type"],
            "params": config["index_params"],
        },
    )
    try:
        col.create_index("video_genre", {"index_type": "INVERTED"})
    except Exception:
        logger.warning("Could not create INVERTED index on video_genre field; scalar filtering may be slow.")

    col.load()
    logger.info(
        "Created Milvus collection=%s dim=%s index=%s metric=%s params=%s",
        collection_name,
        VECTOR_DIM,
        config["index_type"],
        METRIC_TYPE,
        config["index_params"],
    )
    return col


def drop_collection_if_exists(
    algorithm: str | None = None,
    channel: str = "raw.semantic",
) -> bool:
    collection_name = collection_name_for_algorithm(algorithm, channel)
    if not utility.has_collection(collection_name):
        return False
    utility.drop_collection(collection_name)
    logger.warning("Dropped Milvus collection: %s", collection_name)
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
            [str(r.get("youtube_id", "")) for r in records],
            [r["vector"] for r in records],
        ]
    )


def vector_search(
    collection: Collection,
    query_vector: list[float],
    top_k: int = 100,
    algorithm: str | None = None,
    expr: str | None = None,
    include_vector: bool = False,
) -> list[dict]:
    top_k = max(1, int(top_k))
    config = index_config(algorithm)
    if config["index_type"] == "HNSW":
        base_ef = DEEP_SEARCH_EF if top_k >= DEEP_SEARCH_TOP_K else int(SEARCH_PARAMS["ef"])
        search_params = {"ef": max(base_ef, top_k)}
    elif config["index_type"] == "SCANN":
        search_params = {"nprobe": 32, "reorder_k": max(top_k, top_k * 5)}
    else:
        search_params = {}
    output_fields = ["frame_id", "video_id", "frame_number", "timestamp_ms", "image_url", "youtube_id"]
    if include_vector:
        output_fields = output_fields + ["vector"]
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": METRIC_TYPE, "params": search_params},
        limit=top_k,
        expr=expr,
        output_fields=output_fields,
    )
    hits = []
    for hit in results[0]:
        vector = hit.entity.get("vector") if include_vector else None
        hits.append({
            "frame_id":     hit.entity.get("frame_id"),
            "video_id":     hit.entity.get("video_id"),
            "frame_number": hit.entity.get("frame_number"),
            "timestamp_ms": hit.entity.get("timestamp_ms"),
            "image_url":    hit.entity.get("image_url"),
            "youtube_id":   hit.entity.get("youtube_id"),
            "score":        hit.score,
            **({"_vector": list(vector)} if vector is not None else {}),
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
        output_fields=["frame_id", "video_id", "frame_number", "timestamp_ms", "image_url", "youtube_id"],
        limit=max(1, int(limit)),
    )
    rows.sort(key=lambda row: row["timestamp_ms"])
    return [dict(row) for row in rows]


def query_frame_vectors(collection: Collection, frame_ids: list[str]) -> dict[str, list[float]]:
    frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids if frame_id))
    if not frame_ids:
        return {}
    quoted = ", ".join(json.dumps(frame_id) for frame_id in frame_ids)
    rows = query_frame_vector_rows(collection,
        expr=f"frame_id in [{quoted}]",
        output_fields=["frame_id", "vector"],
        limit=len(frame_ids),
    )
    return {
        str(row["frame_id"]): row["vector"].tolist()
        if hasattr(row["vector"], "tolist") else list(row["vector"])
        for row in rows
    }


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
