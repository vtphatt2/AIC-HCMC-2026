"""Offline, verified JPEG fallback for source frames that cannot be seek-decoded.

Decode the original stream sequentially once, select the already indexed PTS,
and publish only pictures whose pre-encoding checksum matches the full frame
map. The archive, selected frame numbers, and embedding vectors are untouched.
"""
from __future__ import annotations

from functools import lru_cache
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.exact_frame_pts import index_path, showinfo_frames
from app.services.readiness_policy import verified_embed_decoder_threads
from app.services.source_timeline import source_fingerprint, source_map_sha256
from app.services.staged_artifacts import file_digest


EXACT_N_FRAME_DIR = (Path(__file__).resolve().parents[3] /
                     "challenge_resources/data/zip_embeddings/exact_n_frames")


def exact_image_path(video_id: str, index, frame_number: int,
                     directory: Path = EXACT_N_FRAME_DIR) -> Path:
    return directory / index_path(video_id, index).stem / f"{frame_number}.jpg"


def _provenance(video_id: str, index, map_path: Path) -> dict:
    return {'version': 1, 'source': source_fingerprint(index),
            'source_map_sha256': source_map_sha256(map_path),
            'decoder_threads': verified_embed_decoder_threads(video_id, index)}


@lru_cache(maxsize=512)
def _read_manifest(path: str, mtime_ns: int, size: int) -> dict:
    return json.loads(Path(path).read_text())


def _manifest(path: Path) -> dict | None:
    try:
        stat = path.stat()
        return _read_manifest(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError):
        return None


def verified_exact_image_bytes(video_id: str, index, frame_number: int, *,
                               directory: Path = EXACT_N_FRAME_DIR) -> bytes | None:
    """Read only a JPEG bound to the current source map and checked file bytes."""
    try:
        path = exact_image_path(video_id, index, frame_number, directory)
        map_path = index_path(video_id, index)
        marker = _manifest(path.parent / 'verified_images.json')
        provenance = _provenance(video_id, index, map_path)
        if not isinstance(marker, dict) or any(
                marker.get(key) != value for key, value in provenance.items()):
            return None
        row = marker.get('images', {}).get(str(frame_number))
        if not isinstance(row, dict):
            return None
        table = np.load(map_path, mmap_mode='r', allow_pickle=False)
        if (table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64 or
                frame_number < 0 or frame_number >= len(table) or
                int(table[frame_number, 0]) != frame_number or
                row.get('source_pts') != int(table[frame_number, 1]) or
                row.get('source_checksum') != int(table[frame_number, 2])):
            return None
        data = path.read_bytes()
        if len(data) != row.get('size') or hashlib.sha256(data).hexdigest() != row.get('sha256'):
            return None
        return data
    except (OSError, ValueError, IndexError, KeyError):
        return None


def verified_exact_image_set(video_id: str, index, frames: list[int], *,
                             directory: Path = EXACT_N_FRAME_DIR) -> bool:
    """Check a complete selected set with one map/manifest read per video."""
    if not frames or len(frames) != len(set(frames)):
        return False
    try:
        map_path = index_path(video_id, index)
        parent = exact_image_path(video_id, index, frames[0], directory).parent
        marker = _manifest(parent / 'verified_images.json')
        provenance = _provenance(video_id, index, map_path)
        if not isinstance(marker, dict) or any(
                marker.get(key) != value for key, value in provenance.items()):
            return False
        records = marker.get('images')
        if not isinstance(records, dict):
            return False
        table = np.load(map_path, mmap_mode='r', allow_pickle=False)
        if table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64:
            return False
        for frame in frames:
            if frame < 0 or frame >= len(table) or int(table[frame, 0]) != frame:
                return False
            record = records.get(str(frame))
            if not isinstance(record, dict) or record.get('source_pts') != int(table[frame, 1]) or \
                    record.get('source_checksum') != int(table[frame, 2]):
                return False
            data = (parent / f'{frame}.jpg').read_bytes()
            if len(data) != record.get('size') or hashlib.sha256(data).hexdigest() != record.get('sha256'):
                return False
        return True
    except (OSError, ValueError, IndexError, KeyError):
        return False


def build_exact_images(video_id: str, index, frames: list[int], *,
                       directory: Path = EXACT_N_FRAME_DIR,
                       timeout: float = 900) -> list[Path]:
    """Serialize publication for one source map across warmer/audit processes."""
    if not frames:
        return []
    parent = exact_image_path(video_id, index, min(frames), directory).parent
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / '.images.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_exact_images_locked(video_id, index, frames, directory=directory,
                                          timeout=timeout)


def _build_exact_images_locked(video_id: str, index, frames: list[int], *,
                               directory: Path, timeout: float) -> list[Path]:
    """Publish exact full-size JPEGs after validating the entire selected batch."""
    if not video_id.startswith("N"):
        raise ValueError("Sequential fallback publication is limited to N videos")
    targets = sorted(set(frames))
    if not targets:
        return []
    table = np.load(index_path(video_id, index), mmap_mode="r", allow_pickle=False)
    if (table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64 or
            not np.array_equal(table[:, 0], np.arange(len(table))) or
            targets[0] < 0 or targets[-1] >= len(table)):
        raise ValueError(f"{video_id}: missing or invalid full decoded-frame map")
    selected = table[targets]
    # A retained picture may share its PTS with an omitted backward-timestamp
    # picture. Select by decoded frame number in that case; the full-map PTS
    # AND pixel checksum below still have to match before any JPEG is exposed.
    unique_pts = len(set(map(int, selected[:, 1]))) == len(targets) and all(
        np.count_nonzero(table[:, 1] == pts) == 1 for pts in selected[:, 1])
    paths = [exact_image_path(video_id, index, frame, directory) for frame in targets]
    map_path = index_path(video_id, index)
    provenance = _provenance(video_id, index, map_path)
    marker_path = paths[0].parent / 'verified_images.json'
    old_marker = _manifest(marker_path)
    if verified_exact_image_set(video_id, index, targets, directory=directory):
        return paths
    directory.mkdir(parents=True, exist_ok=True)
    source = (f"subfile,,start,{index.data_offset},end,"
              f"{index.data_offset + index.video_size},,:{index.zip_path}")
    selection = ("+".join(f"eq(pts\\,{int(pts)})" for pts in selected[:, 1])
                 if unique_pts else
                 "+".join(f"eq(n\\,{frame})" for frame in targets))
    with tempfile.TemporaryDirectory(dir=directory, prefix=".decode-") as scratch:
        scratch_path = Path(scratch)
        command = [
            "/usr/bin/ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
            "-copyts", "-threads", str(verified_embed_decoder_threads(video_id, index)),
            "-i", source, "-map", "0:v:0", "-an",
            "-vf", f"select='{selection}',showinfo", "-vsync", "0",
            "-filter_threads", "1", "-threads:v", "1",
            "-frames:v", str(len(targets)), "-q:v", "4",
            "-start_number", "0", "-f", "image2", str(scratch_path / "%08d.jpg"),
        ]
        with tempfile.TemporaryFile() as diagnostics:
            process = subprocess.run(command, stdout=subprocess.DEVNULL,
                                     stderr=diagnostics, timeout=timeout)
            diagnostics.seek(0)
            records = showinfo_frames(diagnostics.read())
        if process.returncode or len(records) != len(targets):
            raise RuntimeError(f"{video_id}: sequential JPEG decode returned "
                               f"{len(records)}/{len(targets)} frames, exit {process.returncode}")
        for position, (_, pts, checksum) in enumerate(records):
            if pts != int(selected[position, 1]) or checksum != int(selected[position, 2]):
                raise RuntimeError(f"{video_id}/{targets[position]}: sequential pixel verification failed")
            with Image.open(scratch_path / f"{position:08d}.jpg") as image:
                image.verify()
        # No selected image becomes visible until all identities are checked.
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        for position, path in enumerate(paths):
            os.replace(scratch_path / f"{position:08d}.jpg", path)
        records = (dict(old_marker.get('images', {})) if old_marker and all(
            old_marker.get(key) == value for key, value in provenance.items()) else {})
        for frame, path in zip(targets, paths):
            records[str(frame)] = {'source_pts': int(table[frame, 1]),
                                   'source_checksum': int(table[frame, 2]),
                                   'size': path.stat().st_size,
                                   'sha256': file_digest(path)}
        temporary_marker = scratch_path / 'verified_images.json'
        temporary_marker.write_text(json.dumps({**provenance, 'images': records}, sort_keys=True))
        os.replace(temporary_marker, marker_path)
        _read_manifest.cache_clear()
    return paths
