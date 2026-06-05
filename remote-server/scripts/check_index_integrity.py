"""
Validate Milvus collection schema, metadata, duplicates, and image paths.

Run from remote-server:
  python scripts/check_index_integrity.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent
sys.path.insert(0, str(REMOTE_ROOT))

logger = logging.getLogger("check_index_integrity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Milvus video_frames index integrity.")
    parser.add_argument("--sample-root", type=Path, default=REPO_ROOT / "AIC2026_sample")
    parser.add_argument("--static-root", type=Path, default=REMOTE_ROOT / "static")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-missing-images", type=int, default=20)
    return parser.parse_args()


def vector_dim_from_schema(collection: Any) -> int | None:
    for field in collection.schema.fields:
        if field.name == "vector":
            dim = field.params.get("dim")
            return int(dim) if dim is not None else None
    return None


def query_all_rows(collection: Any, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = collection.query(
            expr="frame_number >= 0",
            output_fields=["frame_id", "video_id", "frame_number", "timestamp_ms", "image_url"],
            limit=batch_size,
            offset=offset,
        )
        rows.extend(dict(row) for row in batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return rows


def image_exists(image_url: str, static_root: Path, sample_root: Path) -> bool:
    if not image_url:
        return False

    if image_url.startswith("/static/"):
        relative = image_url.removeprefix("/static/")
        if (static_root / relative).is_file():
            return True
        parts = Path(relative).parts
        if len(parts) >= 3 and parts[0] == "frames":
            sample_path = sample_root / "keyframes" / "keyframes" / Path(*parts[1:])
            return sample_path.is_file()
        return False

    return Path(image_url).is_file()


def validate_rows(
    rows: list[dict[str, Any]],
    static_root: Path,
    sample_root: Path,
    max_missing_images: int,
) -> dict[str, Any]:
    frame_ids = [str(row.get("frame_id", "")) for row in rows]
    duplicates = [frame_id for frame_id, count in Counter(frame_ids).items() if count > 1]

    missing_metadata = []
    missing_images = []
    for row in rows:
        required_missing = [
            field
            for field in ("frame_id", "video_id", "frame_number", "timestamp_ms")
            if row.get(field) in (None, "")
        ]
        if required_missing:
            missing_metadata.append({"frame_id": row.get("frame_id"), "missing": required_missing})

        image_url = str(row.get("image_url") or "")
        if not image_exists(image_url, static_root, sample_root):
            missing_images.append({"frame_id": row.get("frame_id"), "image_url": image_url})

    return {
        "duplicate_frame_ids": duplicates,
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_examples": missing_metadata[:20],
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:max_missing_images],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    from app.db import milvus_client

    milvus_client.connect()
    collection = milvus_client.get_collection()
    dim = vector_dim_from_schema(collection)
    row_count = int(collection.num_entities)
    rows = query_all_rows(collection, max(1, args.batch_size))
    validation = validate_rows(rows, args.static_root, args.sample_root, args.max_missing_images)

    status = "PASS"
    if dim != milvus_client.VECTOR_DIM:
        status = "FAIL"
    if row_count != len(rows):
        status = "FAIL"
    if validation["duplicate_frame_ids"] or validation["missing_metadata_count"]:
        status = "FAIL"
    if validation["missing_image_count"]:
        status = "WARN" if status == "PASS" else status

    print(f"status: {status}")
    print(f"collection: {milvus_client.COLLECTION_NAME}")
    print(f"vector_count: {row_count}")
    print(f"queried_rows: {len(rows)}")
    print(f"dim: {dim}")
    print(f"expected_dim: {milvus_client.VECTOR_DIM}")
    print(f"duplicate_frame_ids: {len(validation['duplicate_frame_ids'])}")
    print(f"missing_metadata_count: {validation['missing_metadata_count']}")
    print(f"missing_image_count: {validation['missing_image_count']}")
    for item in validation["missing_image_examples"]:
        print(f"missing_image: frame_id={item['frame_id']} image_url={item['image_url']}")


if __name__ == "__main__":
    main()
