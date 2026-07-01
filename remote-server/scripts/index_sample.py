"""
Index AIC2026_sample keyframes into PostgreSQL and Milvus.

Run from the repository root or remote-server directory:
  python scripts/index_sample.py --copy-keyframes --recreate-milvus
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent

sys.path.insert(0, str(REMOTE_ROOT))

from scripts.sample_paths import default_sample_root, sample_subdir

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index AIC2026_sample into Milvus/PostgreSQL.")
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=default_sample_root(REPO_ROOT),
        help="Path to AIC2026_sample.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--recreate-milvus",
        action="store_true",
        help="Drop and recreate the selected Milvus collection(s) before inserting vectors.",
    )
    parser.add_argument(
        "--vector-index",
        choices=["hnsw", "flat", "scann", "all"],
        default="hnsw",
        help="Milvus vector index collection to build. Use 'all' to build HNSW, FLAT, and SCANN.",
    )
    parser.add_argument(
        "--copy-keyframes",
        action="store_true",
        help="Copy keyframes into remote-server/static/frames so /static/frames/... works.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect files without writing DBs.")
    return parser.parse_args()


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def youtube_id_from_link(link: str) -> str:
    parsed = urlparse(link)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    return parse_qs(parsed.query).get("v", [""])[0]


def load_metadata(metadata_dir: Path) -> dict[str, dict]:
    metadata = {}
    for path in sorted(metadata_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        video_id = path.stem
        video_link = data.get("video_link", "")
        youtube_id = data.get("youtube_id") or youtube_id_from_link(video_link)
        if not YOUTUBE_ID_PATTERN.fullmatch(str(youtube_id)):
            raise ValueError(f"{path} does not contain a valid YouTube video ID")
        metadata[video_id] = {
            "video_id": video_id,
            "title": data.get("title") or video_id,
            "youtube_id": str(youtube_id),
            "fps": float(data.get("fps") or 25.0),
            "video_link": video_link,
        }
    return metadata


def iter_video_records(sample_root: Path):
    metadata_dir = require_dir(sample_subdir(sample_root, "metadata"), "metadata directory")
    keyframes_dir = require_dir(sample_subdir(sample_root, "keyframes"), "keyframes directory")
    features_dir = require_dir(sample_subdir(sample_root, "PECore-features"), "features directory")
    metadata = load_metadata(metadata_dir)

    for feature_video_dir in sorted(p for p in features_dir.iterdir() if p.is_dir()):
        video_id = feature_video_dir.name
        if video_id == "selected_keyframes":
            continue
        if video_id not in metadata:
            raise FileNotFoundError(f"Metadata file not found for feature directory: {video_id}")

        video_meta = metadata[video_id]
        frame_dir = keyframes_dir / video_id
        npy_paths = sorted(feature_video_dir.glob("*.npy"))
        records = []
        missing_images = 0

        for npy_path in npy_paths:
            frame_stem = npy_path.stem
            image_path = frame_dir / f"{frame_stem}.jpg"
            if not image_path.exists():
                missing_images += 1

            frame_number = int(frame_stem)
            timestamp_ms = int(frame_number / float(video_meta["fps"]) * 1000)
            records.append(
                {
                    "frame_id": f"{video_id}_{frame_stem}",
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "image_url": f"/static/frames/{video_id}/{frame_stem}.jpg",
                    "feature_path": npy_path,
                    "image_path": image_path,
                }
            )

        if records:
            video_meta = {
                **video_meta,
                "duration_ms": max(r["timestamp_ms"] for r in records),
                "frame_count": len(records),
            }
        else:
            video_meta = {**video_meta, "duration_ms": 0, "frame_count": 0}

        yield video_meta, records, missing_images


async def upsert_videos(pool: Any, videos: list[dict]) -> None:
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO videos(video_id, title, youtube_id, fps, duration_ms, frame_count)
            VALUES($1, $2, $3, $4, $5, $6)
            ON CONFLICT(video_id) DO UPDATE SET
                title = EXCLUDED.title,
                youtube_id = EXCLUDED.youtube_id,
                fps = EXCLUDED.fps,
                duration_ms = EXCLUDED.duration_ms,
                frame_count = EXCLUDED.frame_count
            """,
            [
                (
                    v["video_id"],
                    v["title"],
                    v["youtube_id"],
                    float(v["fps"]),
                    int(v["duration_ms"]),
                    int(v["frame_count"]),
                )
                for v in videos
            ],
        )


def copy_keyframes(records: list[dict], static_frames_dir: Path) -> int:
    copied = 0
    for record in tqdm(records, desc="Copying keyframes", unit="frame"):
        src = record["image_path"]
        if not src.exists():
            continue
        dst = static_frames_dir / record["video_id"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied += 1
    return copied


def load_vector(path: Path, expected_dim: int) -> list[float]:
    import numpy as np

    vector = np.load(path).astype("float32").reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(f"{path} has dim {vector.size}; expected {expected_dim}")
    return vector.tolist()


def upsert_vectors(collection: Any, records: list[dict], batch_size: int, vector_dim: int) -> int:
    inserted = 0
    with tqdm(total=len(records), desc="Indexing Milvus vectors", unit="vector") as progress:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            frame_ids = [r["frame_id"] for r in batch]
            video_ids = [r["video_id"] for r in batch]
            frame_numbers = [int(r["frame_number"]) for r in batch]
            timestamps = [int(r["timestamp_ms"]) for r in batch]
            image_urls = [r["image_url"] for r in batch]
            vectors = [load_vector(r["feature_path"], vector_dim) for r in batch]
            collection.upsert([frame_ids, video_ids, frame_numbers, timestamps, image_urls, vectors])
            inserted += len(batch)
            progress.update(len(batch))
            progress.set_postfix(indexed=inserted)
    collection.flush()
    collection.load()
    return inserted


async def main() -> None:
    args = parse_args()
    sample_root = args.sample_root.resolve()
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    videos = []
    all_records = []
    missing_images_total = 0
    for video, records, missing_images in iter_video_records(sample_root):
        videos.append(video)
        all_records.extend(records)
        missing_images_total += missing_images
        print(f"{video['video_id']}: {len(records)} features, {missing_images} missing images")

    print(f"Found {len(videos)} videos and {len(all_records)} feature vectors.")
    if missing_images_total:
        print(f"Warning: {missing_images_total} feature files have no matching .jpg keyframe.")

    if args.dry_run:
        return

    from app.db import milvus_client, postgres_client

    await postgres_client.init_schema()
    pool = await postgres_client.get_pool()
    await upsert_videos(pool, videos)

    milvus_client.connect()

    if args.copy_keyframes:
        copied = copy_keyframes(all_records, REMOTE_ROOT / "static" / "frames")
        print(f"Copied {copied} keyframes into {REMOTE_ROOT / 'static' / 'frames'}")

    target_indexes = ["hnsw", "flat", "scann"] if args.vector_index == "all" else [args.vector_index]
    for vector_index in target_indexes:
        if args.recreate_milvus:
            milvus_client.drop_collection_if_exists(vector_index)
        collection = milvus_client.create_collection_if_missing(vector_index)
        indexed = upsert_vectors(collection, all_records, args.batch_size, milvus_client.VECTOR_DIM)
        collection_name = milvus_client.collection_name_for_algorithm(vector_index)
        print(f"Done. Indexed {indexed} vectors into Milvus collection '{collection_name}' ({vector_index}).")
    await postgres_client.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
