"""
Ingest PE-Core keyframe embeddings from keyframe_pipeline_global_v9_3
"*_results.zip" archives (challenge_resources/data/zip_embeddings/) into Milvus +
PostgreSQL.

Each archive is one organizer zip "lot" (e.g. L26_c_results.zip, produced from
Videos_L26_c.zip) and contains, per video:

  phase1_transnet/video__<video_id>/scenes.json      -- fps, width, height, num_frames
  phase1_transnet/video__<video_id>/keyframes.json   -- selected frame_number list, in
                                                         the same row order as embeddings.npy
  phase2_embeddings/video__<video_id>/embeddings.npy -- (num_keyframes, 1280) float32,
                                                         L2-normalized PE-Core-bigG vectors

No keyframe JPGs ship in these archives (metadata + embeddings only, kept
lightweight for transfer), so image_url is left blank here — every hit
already carries video_id + frame_number/timestamp_ms, which is enough for a
consumer to derive its own thumbnail URL (e.g. local-backend rewrites it to
its own on-demand /api/zip-frame/<video_id>/<timestamp_ms> route, which
decodes that one frame straight from the organizer ZIP). Baking a URL in at
ingest time just risks it going stale the moment a consumer's serving
mechanism changes.

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
VIDEO_DIR_PATTERN = re.compile(r"^videos?__(?P<video_id>.+)$")
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def selected_timestamp_ms(video_id: str, item: dict[str, Any], fps: float,
                          metadata_version: int = 1) -> int:
    """Keep ingest, export and audit on the same source presentation clock."""
    if video_id.startswith("N") and (metadata_version >= 2 or "source_pts" in item):
        if not all(key in item for key in ("source_pts", "source_timebase", "source_checksum")):
            raise ValueError(f"{video_id}: missing verified source identity")
        scale = int(item["source_timebase"])
        if scale <= 0:
            raise ValueError(f"{video_id}: invalid source time base")
        timestamp_ms = (int(item["source_pts"]) * 1000 + scale // 2) // scale
        if metadata_version >= 2 and ("timestamp_ms" not in item or
                                      int(item["timestamp_ms"]) != timestamp_ms):
            raise ValueError(f"{video_id}: selected presentation timestamp differs from source PTS")
        return timestamp_ms
    return int(int(item["frame_number"]) / fps * 1000)


def source_duration_ms(video_id: str, keyframes: dict[str, Any], scenes: dict[str, Any]) -> int:
    """Use the source MP4 track duration for VFR N when the verified field exists."""
    if video_id.startswith('N') and keyframes.get('version', 1) >= 2 and \
            'source_duration_ms' in keyframes:
        duration = keyframes['source_duration_ms']
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise ValueError(f'{video_id}: invalid source duration')
        return duration
    # Old artifacts retain their established nominal-duration behavior.
    return int(int(scenes['num_frames']) / float(scenes['fps']) * 1000)


def youtube_id_from_watch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    return parse_qs(parsed.query).get("v", [""])[0]


def load_media_info(zip_dir: Path) -> dict[str, dict[str, str]]:
    """video_id -> {"youtube_id", "title"} from media-info-*.zip archives
    (one media-info/<video_id>.json per video, with a "watch_url" field)."""
    info: dict[str, dict[str, str]] = {}
    archives = sorted(set(zip_dir.glob("media-info*.zip")) | set(zip_dir.glob("media_info*.zip")))
    # Organizer files use both hyphens and underscores. Resolve aliases only
    # against IDs in the result archives; ambiguous aliases are never guessed.
    canonical_ids: set[str] = set()
    for result in zip_dir.glob("*_results.zip"):
        with zipfile.ZipFile(result) as zf:
            canonical_ids.update(video_id for video_id, _ in video_directories(set(zf.namelist())))
    aliases: dict[str, set[str]] = {}
    for video_id in canonical_ids:
        aliases.setdefault(video_id.replace("_", "-").casefold(), set()).add(video_id)
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                video_id = Path(name).stem
                if canonical_ids and video_id not in canonical_ids:
                    matches = aliases.get(video_id.replace("_", "-").casefold(), set())
                    if len(matches) != 1:
                        if matches:
                            logger.warning("Ambiguous organizer alias %s: %s", video_id, sorted(matches))
                        continue
                    video_id = next(iter(matches))
                data = json.loads(zf.read(name))
                youtube_id = youtube_id_from_watch_url(data.get("watch_url", ""))
                if not YOUTUBE_ID_PATTERN.fullmatch(youtube_id):
                    logger.warning(
                        "%s: watch_url %r has no valid youtube_id, skipping",
                        video_id, data.get("watch_url"),
                    )
                    continue
                item = {"youtube_id": youtube_id, "title": data.get("title") or video_id}
                if video_id in info and info[video_id] != item:
                    raise ValueError(f"Conflicting organizer metadata for {video_id} across media-info archives")
                info[video_id] = item
    if archives:
        logger.info("Loaded youtube_id/title for %s videos from %s media-info archive(s)", len(info), len(archives))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=REPO_ROOT / "challenge_resources" / "data" / "zip_embeddings",
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
    return parser.parse_args()


def iter_result_archives(zip_dir: Path) -> list[Path]:
    archives = sorted(zip_dir.glob("*_results.zip"))
    if not archives:
        raise FileNotFoundError(f"No *_results.zip found under {zip_dir}")
    return archives


def video_directories(names: set[str]) -> list[tuple[str, str]]:
    """Return (video_id, archive directory) for both supported ZIP layouts."""
    directories: dict[str, str] = {}
    for name in names:
        parts = name.split("/")
        if len(parts) != 3 or parts[0] != "phase1_transnet" or parts[2] != "scenes.json":
            continue
        match = VIDEO_DIR_PATTERN.fullmatch(parts[1])
        if match is None:
            continue
        video_id = match.group("video_id")
        previous = directories.setdefault(video_id, parts[1])
        if previous != parts[1]:
            raise ValueError(f"Ambiguous archive directories for {video_id}: {previous}, {parts[1]}")
    return sorted(directories.items())


def iter_video_records(
    zip_path: Path,
    media_info: dict[str, dict[str, str]],
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    import numpy as np
    from app.services.video_quarantine import release_blocked_video_ids

    release_blocked = release_blocked_video_ids()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for video_id, directory in video_directories(names):
            if video_id in release_blocked:
                logger.info("%s: release-blocked source, skipping ingest", video_id)
                continue
            scenes_path = f"phase1_transnet/{directory}/scenes.json"
            keyframes_path = f"phase1_transnet/{directory}/keyframes.json"
            embeddings_path = f"phase2_embeddings/{directory}/embeddings.npy"

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

            known = media_info.get(video_id, {})
            youtube_id = known.get("youtube_id", "")

            records: list[dict[str, Any]] = []
            for row, item in enumerate(selected):
                frame_number = int(item["frame_number"])
                vector = np.asarray(vectors[row], dtype="float32").reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm == 0.0:
                    raise ValueError(f"{video_id} frame {frame_number}: zero-norm vector")
                timestamp_ms = selected_timestamp_ms(
                    video_id, item, fps, keyframes.get("version", 1))
                records.append({
                    "frame_id": f"{video_id}_{frame_number:06d}",
                    "video_id": video_id,
                    "video_genre": "",
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "image_url": "",
                    "youtube_id": youtube_id,
                    "vector": (vector / norm).tolist(),
                })
            video = {
                "video_id": video_id,
                "title": known.get("title") or video_id,
                "youtube_id": known.get("youtube_id", ""),
                "fps": fps,
                "duration_ms": source_duration_ms(video_id, keyframes, scenes),
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

    # Keep only lightweight video metadata globally. Vector records are streamed
    # to Milvus in bounded buffers so the whole embedding dataset never lives
    # in Python memory at once.
    videos: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    total_vectors = 0

    if args.dry_run:
        for archive in archives:
            for video, video_records in tqdm(
                iter_video_records(archive, media_info),
                desc=archive.name,
                unit="video",
            ):
                if video["video_id"] in seen_video_ids:
                    logger.warning("Duplicate video_id %s (already seen), skipping", video["video_id"])
                    continue
                seen_video_ids.add(video["video_id"])
                videos.append(video)
                total_vectors += len(video_records)

        logger.info(
            "Parsed %s videos and %s PE-Core vectors from %s archive(s)",
            len(videos), total_vectors, len(archives),
        )
        return

    from app.db import milvus_client

    milvus_client.connect()

    target_indexes = ["hnsw", "flat", "scann"] if args.vector_index == "all" else [args.vector_index]
    collections: list[tuple[str, Any]] = []
    for vector_index in target_indexes:
        if args.recreate_milvus:
            milvus_client.drop_collection_if_exists(vector_index, "raw.semantic")
        collection = milvus_client.create_collection_if_missing(vector_index, "raw.semantic")
        collections.append((vector_index, collection))

    batch_size = max(1, args.batch_size)
    # A small multiple of the Milvus batch size amortizes Python call/progress
    # overhead while keeping peak RAM bounded. With the default batch_size=256,
    # this holds at most ~1024 vector records plus one video's temporary records.
    stream_buffer_size = batch_size * 120
    pending_records: list[dict[str, Any]] = []
    indexed_by_algorithm = {vector_index: 0 for vector_index in target_indexes}

    def flush_pending() -> None:
        nonlocal pending_records
        if not pending_records:
            return

        batch = pending_records
        pending_records = []

        for vector_index, collection in collections:
            indexed = upsert_vectors(
                collection,
                batch,
                batch_size,
                milvus_client.VECTOR_DIM,
            )
            indexed_by_algorithm[vector_index] += indexed

    for archive in archives:
        for video, video_records in tqdm(
            iter_video_records(archive, media_info),
            desc=archive.name,
            unit="video",
        ):
            if video["video_id"] in seen_video_ids:
                logger.warning("Duplicate video_id %s (already seen), skipping", video["video_id"])
                continue

            seen_video_ids.add(video["video_id"])
            videos.append(video)
            total_vectors += len(video_records)
            pending_records.extend(video_records)

            if len(pending_records) >= stream_buffer_size:
                flush_pending()

    flush_pending()

    logger.info(
        "Parsed %s videos and streamed %s PE-Core vectors from %s archive(s)",
        len(videos), total_vectors, len(archives),
    )

    if not args.skip_postgres:
        await upsert_videos(videos)
        logger.info("Upserted %s videos into PostgreSQL", len(videos))

    for vector_index, _collection in collections:
        collection_name = milvus_client.collection_name_for_algorithm(vector_index, "raw.semantic")
        logger.info(
            "Done. Indexed %s vectors into Milvus collection '%s' (%s).",
            indexed_by_algorithm[vector_index], collection_name, vector_index,
        )


if __name__ == "__main__":
    asyncio.run(main())
