from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


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
    """Rank exact ID, compact ID prefix, then title prefix/substring matches.

    Compact IDs deliberately ignore separators: ``L0_`` becomes ``l0`` and
    therefore finds indexed IDs ``L01_...`` through ``L09_...``. Title search
    is case- and accent-insensitive so operators do not have to type Vietnamese
    diacritics exactly as the organizer metadata does.
    """
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
