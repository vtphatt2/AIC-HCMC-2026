"""
Ingest PE-Core frame embeddings from AIC2026_sample into Milvus.

This script indexes visual embeddings and upserts minimal video metadata needed
by the backend result formatter. It does not create OCR or ASR data.

Run from remote-server:
  python scripts/ingest_embeddings_to_milvus.py --copy-keyframes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Iterator
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

logger = logging.getLogger("ingest_embeddings_to_milvus")
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest AIC2026_sample PE-Core embeddings to Milvus.")
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=default_sample_root(REPO_ROOT),
        help="Path to AIC2026_sample. Defaults to the first existing repo-local or sibling dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--copy-keyframes",
        action="store_true",
        help="Copy keyframe jpgs into remote-server/static/frames for /static/frames URLs.",
    )
    parser.add_argument(
        "--vector-index",
        choices=["hnsw", "flat", "scann", "all"],
        default="hnsw",
        help="Milvus vector index collection to build. Use 'all' to build HNSW, FLAT, and SCANN.",
    )
    parser.add_argument(
        "--recreate-milvus",
        action="store_true",
        help="Drop and recreate the selected Milvus collection(s) before inserting vectors.",
    )
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Only ingest Milvus vectors; do not upsert video metadata to PostgreSQL.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--features-subdir",
        type=str,
        default="PECore-features",
        help="Subdirectory name under sample-root containing .npy vector files (default: PECore-features).",
    )
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


def load_metadata(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(metadata_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        video_id = path.stem
        youtube_id = data.get("youtube_id") or youtube_id_from_link(data.get("video_link", ""))
        if not YOUTUBE_ID_PATTERN.fullmatch(str(youtube_id)):
            raise ValueError(f"{path} does not contain a valid YouTube video ID")
        metadata[video_id] = {
            "video_id": video_id,
            "title": data.get("title") or video_id,
            "youtube_id": str(youtube_id),
            "fps": float(data.get("fps") or 25.0),
        }
    return metadata


def iter_video_records(sample_root: Path, features_subdir: str = "PECore-features") -> Iterator[tuple[dict[str, Any], list[dict[str, Any]], int]]:
    metadata_dir = require_dir(sample_subdir(sample_root, "metadata"), "metadata directory")
    keyframes_dir = require_dir(sample_subdir(sample_root, "keyframes"), "keyframes directory")
    features_dir = require_dir(sample_subdir(sample_root, features_subdir), f"features directory ({features_subdir})")
    metadata = load_metadata(metadata_dir)

    for feature_video_dir in sorted(path for path in features_dir.iterdir() if path.is_dir()):
        video_id = feature_video_dir.name
        if video_id == "selected_keyframes":
            continue
        if video_id not in metadata:
            raise FileNotFoundError(f"Metadata file not found for feature directory: {video_id}")

        fps = float(metadata[video_id]["fps"])
        frame_dir = keyframes_dir / video_id
        records: list[dict[str, Any]] = []
        missing_images = 0

        for feature_path in sorted(feature_video_dir.glob("*.npy")):
            frame_stem = feature_path.stem
            frame_number = int(frame_stem)
            timestamp_ms = int(frame_number / fps * 1000)
            image_path = frame_dir / f"{frame_stem}.jpg"
            if not image_path.exists():
                missing_images += 1

            records.append(
                {
                    "frame_id": f"{video_id}_{frame_stem}",
                    "video_id": video_id,
                    "video_genre": "",
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "image_url": f"/static/frames/{video_id}/{frame_stem}.jpg",
                    "feature_path": feature_path,
                    "image_path": image_path,
                }
            )

        video = {
            "video_id": video_id,
            "title": metadata[video_id]["title"],
            "youtube_id": metadata[video_id]["youtube_id"],
            "fps": fps,
            "duration_ms": max((r["timestamp_ms"] for r in records), default=0),
            "frame_count": len(records),
        }
        yield video, records, missing_images


def load_vector(path: Path, expected_dim: int) -> list[float]:
    import numpy as np

    vector = np.load(path).astype("float32").reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(f"{path} has dim {vector.size}; expected {expected_dim}")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{path} has zero-norm vector")
    return (vector / norm).tolist()


def copy_keyframes(records: list[dict[str, Any]], static_frames_dir: Path) -> int:
    copied = 0
    for record in tqdm(records, desc="Copying keyframes", unit="frame"):
        src = Path(record["image_path"])
        if not src.is_file():
            continue
        dst = static_frames_dir / str(record["video_id"]) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied += 1
    return copied


async def upsert_videos(videos: list[dict[str, Any]]) -> None:
    from app.db import postgres_client

    await postgres_client.init_schema()
    pool = await postgres_client.get_pool()
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
                    video["video_id"],
                    video["title"],
                    video["youtube_id"],
                    float(video["fps"]),
                    int(video["duration_ms"]),
                    int(video["frame_count"]),
                )
                for video in videos
            ],
        )
    await postgres_client.close_pool()


def upsert_vectors(collection: Any, records: list[dict[str, Any]], batch_size: int, vector_dim: int) -> int:
    from app.db import milvus_client

    indexed = 0
    starts = range(0, len(records), batch_size)
    with tqdm(total=len(records), desc="Indexing Milvus vectors", unit="vector") as progress:
        for start in starts:
            batch = []
            for record in records[start : start + batch_size]:
                batch.append({**record, "vector": load_vector(Path(record["feature_path"]), vector_dim)})
            milvus_client.upsert_frame_vectors(collection, batch)
            indexed += len(batch)
            progress.update(len(batch))
            progress.set_postfix(indexed=indexed)
            logger.info("Indexed %s/%s vectors", indexed, len(records))

    collection.flush()
    collection.load()
    return indexed


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    sample_root = args.sample_root.resolve()
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    videos: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    missing_images_total = 0
    for video, video_records, missing_images in iter_video_records(sample_root, args.features_subdir):
        videos.append(video)
        records.extend(video_records)
        missing_images_total += missing_images
        logger.info("%s: %s vectors, %s missing keyframes", video["video_id"], len(video_records), missing_images)

    logger.info("Found %s videos and %s PE-Core vectors under %s", len(videos), len(records), sample_root)
    if missing_images_total:
        logger.warning("%s vectors have no matching .jpg keyframe", missing_images_total)
    if args.dry_run:
        return

    from app.db import milvus_client

    milvus_client.connect()

    if not args.skip_postgres:
        await upsert_videos(videos)
        logger.info("Upserted %s videos into PostgreSQL", len(videos))

    # Hydrate video_genre from PostgreSQL (if indexed by index_transcripts.py prior)
    if not args.skip_postgres:
        try:
            from app.db import postgres_client as _pg
            genre_rows = await _pg.fetch_video_metadata([v["video_id"] for v in videos])
            genre_map = {r["video_id"]: r.get("genre", "") or "" for r in genre_rows}
            for record in records:
                record["video_genre"] = genre_map.get(record["video_id"], "")
            logger.info("Hydrated video_genre for %d videos from PostgreSQL", len(genre_map))
        except Exception as exc:
            logger.warning("Could not hydrate video_genre: %s", exc)

    if args.copy_keyframes:
        copied = copy_keyframes(records, REMOTE_ROOT / "static" / "frames")
        logger.info("Copied %s keyframes into %s", copied, REMOTE_ROOT / "static" / "frames")

    target_indexes = ["hnsw", "flat", "scann"] if args.vector_index == "all" else [args.vector_index]
    for vector_index in target_indexes:
        if args.recreate_milvus:
            milvus_client.drop_collection_if_exists(vector_index)
        collection = milvus_client.create_collection_if_missing(vector_index)
        indexed = upsert_vectors(collection, records, max(1, args.batch_size), milvus_client.VECTOR_DIM)
        collection_name = milvus_client.collection_name_for_algorithm(vector_index)
        logger.info(
            "Done. Indexed %s vectors into Milvus collection '%s' (%s).",
            indexed,
            collection_name,
            vector_index,
        )


if __name__ == "__main__":
    asyncio.run(main())
