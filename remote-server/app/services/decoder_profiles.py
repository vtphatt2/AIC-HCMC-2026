"""Source-bound decoder settings promoted only after complete independent replay."""
from __future__ import annotations

from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / 'challenge_resources/data/source_decoder_profiles.json'


def profile_path() -> Path:
    return Path(os.getenv('SOURCE_DECODER_PROFILES_PATH') or DEFAULT_PATH)


@lru_cache(maxsize=4)
def _read(path: str, mtime_ns: int, size: int) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get('version') != 1 or not isinstance(data.get('profiles'), dict):
        raise ValueError(f'Invalid decoder profile manifest: {path}')
    return data


def source_identity(index) -> dict:
    stat = index.zip_path.stat()
    return {'path': str(index.zip_path.resolve()), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'offset': index.data_offset, 'video_size': index.video_size}


def validate_replay(evidence: dict, source: dict, threads: int) -> None:
    base = evidence.get('expected_time_base')
    if (type(threads) is not int or not 1 <= threads <= 16 or
            not isinstance(base, list) or len(base) != 2 or
            any(type(value) is not int or value <= 0 for value in base) or
            evidence.get('status') != 'exact' or evidence.get('source') != source or
            evidence.get('decoder_threads') != threads or evidence.get('ffmpeg_exit') != 0 or
            type(evidence.get('source_map_rows')) is not int or evidence['source_map_rows'] < 1 or
            evidence.get('decoded_rows') != evidence['source_map_rows'] or
            evidence.get('pts_mismatch_count') != 0 or evidence.get('checksum_mismatch_count') != 0 or
            evidence.get('time_bases') != [evidence.get('expected_time_base')]):
        raise ValueError('Decoder profile requires complete matching source/PTS/checksum/time-base replay')


@lru_cache(maxsize=64)
def _verify_map(path: str, mtime_ns: int, size: int, digest: str) -> None:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError('Verified decoder source map checksum changed')


def decoder_profile(video_id: str, index) -> dict | None:
    path = profile_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    profile = _read(str(path.resolve()), stat.st_mtime_ns, stat.st_size)['profiles'].get(video_id)
    if profile is None:
        return None
    if not isinstance(profile, dict) or profile.get('source') != source_identity(index):
        raise ValueError(f'{video_id}: source changed; decoder profile needs verification')
    evidence = profile.get('verification', {})
    if evidence.get('video_id') != video_id:
        raise ValueError(f'{video_id}: decoder profile evidence belongs to another video')
    validate_replay(evidence, profile['source'], profile.get('threads'))
    if evidence['expected_time_base'] != [1, index.timescale]:
        raise ValueError(f'{video_id}: decoder profile time base differs from source track')
    source_map = Path(evidence['map_path'])
    map_stat = source_map.stat()
    if map_stat.st_size != evidence['map_size'] or map_stat.st_mtime_ns != evidence['map_mtime_ns']:
        raise ValueError(f'{video_id}: verified source map changed; decoder profile needs verification')
    _verify_map(str(source_map), map_stat.st_mtime_ns, map_stat.st_size, profile['map_sha256'])
    return profile


def register_profile(video_id: str, index, evidence: dict, report_path: Path) -> dict:
    """Register offline evidence; this never changes release/quarantine state."""
    import numpy as np
    source = source_identity(index)
    threads = evidence.get('decoder_threads')
    validate_replay(evidence, source, threads)
    if evidence['expected_time_base'] != [1, index.timescale]:
        raise ValueError('Replay time base differs from source track')
    if evidence.get('video_id') != video_id:
        raise ValueError('Replay belongs to a different video')
    from .exact_frame_pts import index_path
    map_path = index_path(video_id, index, decoder_threads=threads)
    stat = map_path.stat()
    if (str(map_path) != evidence.get('map_path') or stat.st_size != evidence.get('map_size') or
            stat.st_mtime_ns != evidence.get('map_mtime_ns')):
        raise ValueError('Source map changed after independent replay')
    table = np.load(map_path, mmap_mode='r', allow_pickle=False)
    if (table.shape != (evidence['source_map_rows'], 3) or table.dtype != np.int64 or
            not np.array_equal(table[:, 0], np.arange(len(table)))):
        raise ValueError('Replay map does not contain every source frame identity')
    profile = {'threads': threads, 'source': source, 'verification': evidence,
               'map_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
               'evidence_path': str(report_path.resolve()),
               'verification_sha256': hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()}
    path = profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'version': 1, 'profiles': {}}
        if data.get('version') != 1 or not isinstance(data.get('profiles'), dict):
            raise ValueError('Invalid existing decoder profile manifest')
        data['profiles'][video_id] = profile
        temporary = path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        os.replace(temporary, path)
    return profile
