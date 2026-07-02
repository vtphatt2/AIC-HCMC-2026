"""
Populate the PostgreSQL `videos` table with youtube_id from metadata JSONs.

This is a lightweight script that only reads metadata/*.json and upserts
into the `videos` table. It does NOT touch Milvus.

Run from remote-server:
    python scripts/upsert_video_metadata.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REMOTE_ROOT.parent
sys.path.insert(0, str(REMOTE_ROOT))

from scripts.sample_paths import default_sample_root, sample_subdir

logger = logging.getLogger("upsert_video_metadata")
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def youtube_id_from_link(link: str) -> str:
    if not link:
        return ""
    try:
        parsed = urlparse(link)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/")
        return parse_qs(parsed.query).get("v", [""])[0]
    except Exception:
        return ""


def load_video_metadata(sample_root: Path) -> list[dict]:
    metadata_dir = sample_subdir(sample_root, "metadata")
    if not metadata_dir.is_dir():
        logger.error("Metadata directory not found: %s", metadata_dir)
        return []

    videos = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            video_id = path.stem
            youtube_id = data.get("youtube_id") or youtube_id_from_link(data.get("video_link", ""))
            if youtube_id and not YOUTUBE_ID_PATTERN.fullmatch(youtube_id):
                logger.warning("%s: invalid youtube_id '%s', skipping", path.name, youtube_id)
                continue
            videos.append({
                "video_id": video_id,
                "title": data.get("title") or video_id,
                "youtube_id": youtube_id or "",
                "fps": float(data.get("fps") or 25.0),
            })
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path.name, exc)
    return videos


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(REMOTE_ROOT / ".env", override=True)
    os.environ.setdefault("ENV_MODE", "SERVER")

    sample_root = default_sample_root(REPO_ROOT)
    videos = load_video_metadata(sample_root)
    if not videos:
        logger.error("No video metadata found")
        sys.exit(1)

    logger.info("Found %d video metadata entries", len(videos))
    for v in videos:
        logger.info("  %s → youtube_id=%s", v["video_id"], v["youtube_id"])

    from app.db import postgres_client
    await postgres_client.init_schema()
    upserted = await postgres_client.upsert_video_metadata(videos)
    logger.info("Upserted %d videos into PostgreSQL", upserted)
    await postgres_client.close_pool()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
