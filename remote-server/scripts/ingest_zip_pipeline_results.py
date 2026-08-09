"""
Ingest PE-Core keyframe embeddings from keyframe_pipeline_global_v9_3
"*_results.zip" archives (challenge_resources/data/zip_file/) into Milvus +
PostgreSQL.

Each archive is one organizer zip "lot" (e.g. L26_c_results.zip, produced from
Videos_L26_c.zip) and contains, per video:

  phase1_transnet/video__<video_id>/scenes.json      -- fps, width, height, num_frames
  phase1_transnet/video__<video_id>/keyframes.json   -- selected frame_number list, in
                                                         the same row order as embeddings.npy
  phase2_embeddings/video__<video_id>/embeddings.npy -- (num_keyframes, 1280) float32,
                                                         L2-normalized PE-Core-bigG vectors

No keyframe JPGs ship in these archives (metadata + embeddings only, kept
lightweight for transfer). image_url is left pointing at the zip-video
streaming proxy (/api/zip-video/<video_id>) as a placeholder until a frame
thumbnail route exists for this source.

youtube_id/title come from the organizers' media-info archive (one
media-info/<video_id>.json per video, with a "watch_url"), e.g.
media-info-aic25-b1.zip under the same --zip-dir. YouTube stays the primary
player everywhere in the app (VideoModal tries it first whenever a video has
a youtube_id) — this script does not opt out of that; it just has nothing to
set for a video_id missing from the media-info archive, and playback for
those falls back to the zip-video proxy like any other video with no known
youtube_id would.

Run from remote-server:
  python scripts/ingest_zip_pipeline_results.py --dry-run
  python scripts/ingest_zip_pipeline_results.py
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import re
import sys
import zipfile
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

from scripts.ingest_embeddings_to_milvus import upsert_vectors, upsert_videos  # noqa: E402

logger = logging.getLogger("ingest_zip_pipeline_results")
VIDEO_DIR_PATTERN = re.compile(r"^video__(?P<video_id>.+)$")
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def youtube_id_from_watch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    return parse_qs(parsed.query).get("v", [""])[0]


def load_media_info(zip_dir: Path) -> dict[str, dict[str, str]]:
    """video_id -> {"youtube_id", "title"} from media-info-*.zip archives
    (one media-info/<video_id>.json per video, with a "watch_url" field)."""
    info: dict[str, dict[str, str]] = {}
    archives = sorted(zip_dir.glob("media-info*.zip"))
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                video_id = Path(name).stem
                data = json.loads(zf.read(name))
                youtube_id = youtube_id_from_watch_url(data.get("watch_url", ""))
                if not YOUTUBE_ID_PATTERN.fullmatch(youtube_id):
                    logger.warning(
                        "%s: watch_url %r has no valid youtube_id, skipping",
                        video_id, data.get("watch_url"),
                    )
                    continue
                info[video_id] = {"youtube_id": youtube_id, "title": data.get("title") or video_id}
    if archives:
        logger.info("Loaded youtube_id/title for %s videos from %s media-info archive(s)", len(info), len(archives))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=REPO_ROOT / "challenge_resources" / "data" / "zip_file",
        help="Directory containing *_results.zip archives.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
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
        "--image-url-template",
        default="/api/zip-video/{video_id}",
        help="image_url stored per frame (placeholder until a zip-source thumbnail "
             "route exists). {video_id} is substituted.",
    )
    return parser.parse_args()


def iter_result_archives(zip_dir: Path) -> list[Path]:
    archives = sorted(zip_dir.glob("*_results.zip"))
    if not archives:
        raise FileNotFoundError(f"No *_results.zip found under {zip_dir}")
    return archives


def iter_video_records(
    zip_path: Path,
    image_url_template: str,
    media_info: dict[str, dict[str, str]],
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    import numpy as np

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        video_ids = sorted({
            match.group("video_id")
            for name in names
            if name.startswith("phase1_transnet/")
            for match in [VIDEO_DIR_PATTERN.match(name.split("/", 2)[1])]
            if match
        })

        for video_id in video_ids:
            scenes_path = f"phase1_transnet/video__{video_id}/scenes.json"
            keyframes_path = f"phase1_transnet/video__{video_id}/keyframes.json"
            embeddings_path = f"phase2_embeddings/video__{video_id}/embeddings.npy"

            if embeddings_path not in names:
                logger.warning("%s: no embeddings.npy in %s, skipping", video_id, zip_path.name)
                continue

            scenes = json.loads(zf.read(scenes_path))
            keyframes = json.loads(zf.read(keyframes_path))
            fps = float(scenes["fps"])
            selected = keyframes["keyframes"]

            vectors = np.load(io.BytesIO(zf.read(embeddings_path)))
            if vectors.shape[0] != len(selected):
                raise ValueError(
                    f"{video_id}: {vectors.shape[0]} embedding rows but {len(selected)} "
                    f"selected keyframes (row order must match 1:1)"
                )

            records: list[dict[str, Any]] = []
            for row, item in enumerate(selected):
                frame_number = int(item["frame_number"])
                vector = np.asarray(vectors[row], dtype="float32").reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm == 0.0:
                    raise ValueError(f"{video_id} frame {frame_number}: zero-norm vector")
                records.append({
                    "frame_id": f"{video_id}_{frame_number:06d}",
                    "video_id": video_id,
                    "video_genre": "",
                    "frame_number": frame_number,
                    "timestamp_ms": int(frame_number / fps * 1000),
                    "image_url": image_url_template.format(video_id=video_id),
                    "vector": (vector / norm).tolist(),
                })

            known = media_info.get(video_id, {})
            video = {
                "video_id": video_id,
                "title": known.get("title") or video_id,
                "youtube_id": known.get("youtube_id", ""),
                "fps": fps,
                "duration_ms": int(int(scenes["num_frames"]) / fps * 1000),
                "frame_count": len(records),
            }
            yield video, records


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    load_dotenv(REMOTE_ROOT / ".env", override=True)

    archives = iter_result_archives(args.zip_dir)
    logger.info("Found %s result archive(s) under %s", len(archives), args.zip_dir)
    media_info = load_media_info(args.zip_dir)

    videos: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()

    for archive in archives:
        for video, video_records in tqdm(
            iter_video_records(archive, args.image_url_template, media_info),
            desc=archive.name,
            unit="video",
        ):
            if video["video_id"] in seen_video_ids:
                logger.warning("Duplicate video_id %s (already seen), skipping", video["video_id"])
                continue
            seen_video_ids.add(video["video_id"])
            videos.append(video)
            records.extend(video_records)

    logger.info("Parsed %s videos and %s PE-Core vectors from %s archive(s)", len(videos), len(records), len(archives))
    if args.dry_run:
        return

    from app.db import milvus_client

    milvus_client.connect()

    if not args.skip_postgres:
        await upsert_videos(videos)
        logger.info("Upserted %s videos into PostgreSQL", len(videos))

    target_indexes = ["hnsw", "flat", "scann"] if args.vector_index == "all" else [args.vector_index]
    for vector_index in target_indexes:
        if args.recreate_milvus:
            milvus_client.drop_collection_if_exists(vector_index, "raw.semantic")
        collection = milvus_client.create_collection_if_missing(vector_index, "raw.semantic")
        indexed = upsert_vectors(collection, records, max(1, args.batch_size), milvus_client.VECTOR_DIM)
        collection_name = milvus_client.collection_name_for_algorithm(vector_index, "raw.semantic")
        logger.info(
            "Done. Indexed %s vectors into Milvus collection '%s' (%s).",
            indexed, collection_name, vector_index,
        )


if __name__ == "__main__":
    asyncio.run(main())
