import os
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "video_frames")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1280"))  # PE-Core-bigG-14-448 produces 1280-dim


def connect():
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
    )


def get_collection() -> Collection:
    col = Collection(COLLECTION_NAME)
    col.load()
    return col


def create_collection_if_missing() -> Collection:
    """Create the video_frames collection with an HNSW index if it does not exist."""
    if utility.has_collection(COLLECTION_NAME):
        return get_collection()

    fields = [
        FieldSchema(name="frame_id",     dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="video_id",     dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="frame_number", dtype=DataType.INT64),
        FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
        FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(fields, description="Frame-level visual embeddings (PE-Core-bigG-14-448)")
    col = Collection(COLLECTION_NAME, schema)

    col.create_index(
        "vector",
        {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        },
    )
    col.load()
    return col


def vector_search(collection: Collection, query_vector: list[float], top_k: int = 100) -> list[dict]:
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["frame_id", "video_id", "frame_number", "timestamp_ms"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "frame_id":     hit.entity.get("frame_id"),
            "video_id":     hit.entity.get("video_id"),
            "frame_number": hit.entity.get("frame_number"),
            "timestamp_ms": hit.entity.get("timestamp_ms"),
            "score":        hit.score,  # cosine similarity, higher = better
        })
    return hits
