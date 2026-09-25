"""Reversible search visibility for audited problem videos; never deletes data."""
from __future__ import annotations

import json
import os
import fcntl
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / 'challenge_resources/data/video_quarantine.json'


def manifest_path() -> Path:
    return Path(os.getenv('VIDEO_QUARANTINE_PATH') or DEFAULT_PATH)


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
    path = manifest_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return _read(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def block_release(video_id: str, *, reason: str, details: str, evidence: str) -> None:
    """Atomically retain a source while preventing release and search use."""
    if not video_id.startswith('N') or not all(
            isinstance(value, str) and value.strip()
            for value in (reason, details, evidence)):
        raise ValueError('A release block requires an N video and complete evidence')
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + '.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'version': 1, 'videos': {}}
        if data.get('version') != 1 or not isinstance(data.get('videos'), dict):
            raise ValueError(f'Invalid video quarantine manifest: {path}')
        previous = data['videos'].get(video_id, {})
        data['videos'][video_id] = {
            **previous, 'reason': reason, 'details': details, 'evidence': evidence,
            'scope': 'search_only; original video and preprocessing preserved',
            'release_blocked': True,
        }
        data['videos'][video_id].pop('released', None)
        data['videos'][video_id].pop('release_evidence', None)
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
        _read.cache_clear()


def clear_release_block(video_id: str, *, reason: str, details: str,
                        evidence: str, expected_reason: str) -> None:
    """Clear only the audited release block, retaining other quarantine scope."""
    path = manifest_path()
    lock_path = path.with_suffix(path.suffix + '.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        entry = data.get('videos', {}).get(video_id)
        if (isinstance(entry, dict) and entry.get('reason') == reason and
                entry.get('release_blocked') is not True):
            return
        if (not isinstance(entry, dict) or entry.get('reason') != expected_reason or
                entry.get('release_blocked') is not True):
            raise ValueError(f'{video_id}: release block changed; refusing automatic clearance')
        entry.update(reason=reason, details=details, evidence=evidence)
        entry.pop('release_blocked')
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
        _read.cache_clear()


def release_videos(expected_reasons: dict[str, str], *, evidence: str) -> None:
    """Atomically make a fully audited N generation visible.

    The caller supplies the reasons it audited so a changed quarantine entry
    cannot be released accidentally. Source-identity blocks must be cleared
    after candidate validation and before the final live audit.
    """
    if (not expected_reasons or
            not isinstance(evidence, str) or not evidence.strip() or
            any(not video.startswith('N') or
                not isinstance(reason, str) or not reason.strip()
                for video, reason in expected_reasons.items())):
        raise ValueError('A verified release requires N videos, expected reasons, and evidence')
    path = manifest_path()
    lock_path = path.with_suffix(path.suffix + '.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        if data.get('version') != 1 or not isinstance(data.get('videos'), dict):
            raise ValueError(f'Invalid video quarantine manifest: {path}')
        for video, expected_reason in expected_reasons.items():
            entry = data['videos'].get(video)
            if (not isinstance(entry, dict) or
                    entry.get('reason') != expected_reason or
                    entry.get('release_blocked') is True):
                raise ValueError(f'{video}: quarantine state changed or remains release-blocked')
        for video in expected_reasons:
            data['videos'][video]['released'] = True
            data['videos'][video]['release_evidence'] = evidence
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
        _read.cache_clear()


def excluded_video_ids() -> frozenset[str]:
    return frozenset(video for video, entry in _entries().items()
                     if entry.get('released') is not True)


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
