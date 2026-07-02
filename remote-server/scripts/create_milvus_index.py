"""
Create the Milvus collection/index used by PE-Core visual retrieval.

Run from remote-server:
  python scripts/create_milvus_index.py
  python scripts/create_milvus_index.py --recreate
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REMOTE_ROOT))

logger = logging.getLogger("create_milvus_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Milvus vector index collection(s).")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection before creating it.",
    )
    parser.add_argument(
        "--vector-index",
        choices=["hnsw", "flat", "scann", "all"],
        default="hnsw",
        help="Milvus vector index collection to create. Use 'all' to create HNSW, FLAT, and SCANN.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    from app.db import milvus_client

    args = parse_args()
    milvus_client.connect()
    target_indexes = ["hnsw", "flat", "scann"] if args.vector_index == "all" else [args.vector_index]
    for vector_index in target_indexes:
        if args.recreate:
            milvus_client.drop_collection_if_exists(vector_index)

        collection = milvus_client.create_collection_if_missing(vector_index)
        config = milvus_client.index_config(vector_index)
        logger.info(
            "Ready: collection=%s dim=%s metric=%s index=%s params=%s rows=%s",
            config["collection"],
            milvus_client.VECTOR_DIM,
            milvus_client.METRIC_TYPE,
            config["index_type"],
            config["index_params"],
            collection.num_entities,
        )


if __name__ == "__main__":
    main()
