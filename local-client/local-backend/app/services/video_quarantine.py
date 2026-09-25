"""Read the shared release blocklist before exposing local NumPy or ZIP data."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parents[4] / 'challenge_resources/data/video_quarantine.json'


@lru_cache(maxsize=4)
def _read(path: str, mtime_ns: int, size: int) -> frozenset[str]:
    data = json.loads(Path(path).read_text())
    if data.get('version') != 1 or not isinstance(data.get('videos'), dict):
        raise ValueError(f'Invalid video quarantine manifest: {path}')
    return frozenset(video for video, entry in data['videos'].items()
                     if isinstance(entry, dict) and entry.get('release_blocked') is True)


def release_blocked_video_ids() -> frozenset[str]:
    path = Path(os.getenv('VIDEO_QUARANTINE_PATH') or DEFAULT_PATH)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return frozenset()
    return _read(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def filter_release_rows(rows: list[dict]) -> list[dict]:
    blocked = release_blocked_video_ids()
    return [row for row in rows if str(row.get('video_id', '')) not in blocked]
