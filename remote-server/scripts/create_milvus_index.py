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
    parser = argparse.ArgumentParser(description="Create Milvus HNSW index for video_frames.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection before creating it.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    from app.db import milvus_client

    args = parse_args()
    milvus_client.connect()
    if args.recreate:
        milvus_client.drop_collection_if_exists()

    collection = milvus_client.create_collection_if_missing()
    logger.info(
        "Ready: collection=%s dim=%s metric=%s index=%s M=%s efConstruction=%s search_ef=%s rows=%s",
        milvus_client.COLLECTION_NAME,
        milvus_client.VECTOR_DIM,
        milvus_client.METRIC_TYPE,
        milvus_client.INDEX_TYPE,
        milvus_client.INDEX_PARAMS["M"],
        milvus_client.INDEX_PARAMS["efConstruction"],
        milvus_client.SEARCH_PARAMS["ef"],
        collection.num_entities,
    )


if __name__ == "__main__":
    main()
