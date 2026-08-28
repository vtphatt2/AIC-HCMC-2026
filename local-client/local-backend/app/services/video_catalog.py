from __future__ import annotations

import re
import unicodedata
import json
import os
import zipfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


def _search_text(value: object) -> str:
    text = str(value or "").replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(without_accents.split())


def _compact_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", _search_text(value))


def search_video_catalog(
    videos: Iterable[dict],
    query: str,
    limit: int = 12,
) -> list[dict]:
    """Rank exact ID, compact ID prefix, then title prefix/substring matches."""
    query_text = _search_text(query)
    query_id = _compact_id(query)
    if not query_text and not query_id:
        return []

    ranked: list[tuple[int, str, dict]] = []
    for video in videos:
        video_id = str(video.get("video_id") or "")
        title = str(video.get("title") or video_id)
        compact_id = _compact_id(video_id)
        normalized_title = _search_text(title)

        if query_id and compact_id == query_id:
            rank = 0
        elif query_id and compact_id.startswith(query_id):
            rank = 1
        elif query_text and normalized_title == query_text:
            rank = 2
        elif query_text and normalized_title.startswith(query_text):
            rank = 3
        elif query_text and query_text in normalized_title:
            rank = 4
        else:
            continue

        result = dict(video)
        result["video_id"] = video_id
        result["title"] = title
        ranked.append((rank, video_id.casefold(), result))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[: max(1, min(int(limit), 50))]]


@lru_cache(maxsize=1)
def indexed_video_catalog() -> list[dict]:
    """Join local NumPy index IDs with titles from organizer media-info ZIPs."""
    from app.db import numpy_vector_store

    configured = os.getenv("RESULTS_ZIP_DIR", "").strip()
    zip_dir = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[4]
        / "challenge_resources"
        / "data"
        / "zip_embeddings"
    )
    titles: dict[str, str] = {}
    for archive_path in sorted(zip_dir.glob("media-info*.zip")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    if not name.endswith(".json"):
                        continue
                    video_id = Path(name).stem
                    payload = json.loads(archive.read(name))
                    titles[video_id] = str(payload.get("title") or video_id)
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
            continue

    return [
        {"video_id": video_id, "title": titles.get(video_id, video_id)}
        for video_id in numpy_vector_store.video_ids()
    ]
