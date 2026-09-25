"""Reversible search visibility for audited problem videos; never deletes data."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / 'challenge_resources/data/video_quarantine.json'


@lru_cache(maxsize=4)
def _read(path: str, mtime_ns: int, size: int) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    if data.get('version') != 1 or not isinstance(data.get('videos'), dict):
        raise ValueError(f'Invalid video quarantine manifest: {path}')
    for video, entry in data['videos'].items():
        if not video or not isinstance(entry, dict) or not entry.get('reason'):
            raise ValueError(f'Missing quarantine reason for {video}')
    return data['videos']


def _entries() -> dict[str, dict]:
    path = Path(os.getenv('VIDEO_QUARANTINE_PATH') or DEFAULT_PATH)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return _read(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def excluded_video_ids() -> frozenset[str]:
    return frozenset(_entries())


def release_blocked_video_ids() -> frozenset[str]:
    """Videos with unresolved source identity that offline jobs must skip."""
    return frozenset(video for video, entry in _entries().items()
                     if entry.get('release_blocked') is True)


def milvus_expr(existing: str | None = None) -> str | None:
    blocked = excluded_video_ids()
    if not blocked:
        return existing
    clause = 'video_id not in ' + json.dumps(sorted(blocked), ensure_ascii=True)
    return f'({existing}) and ({clause})' if existing else clause


def filter_rows(rows):
    blocked = excluded_video_ids()
    return [row for row in rows if str(row.get('video_id', '')) not in blocked]
