"""Small, source-identified maps from decoded frame indices to presentation PTS.

MOV sample tables can contain pictures FFmpeg does not output, and edit lists can
shift timestamps.  Search frame numbers refer to FFmpeg's decoded output, so a
sample-table index alone is not sufficient for an exact random-access image.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .readiness_policy import source_map_decoder_threads


REPO_ROOT = Path(__file__).resolve().parents[3]
PTS_DIR = REPO_ROOT / "challenge_resources/data/zip_embeddings/frame_pts"
_FRAME_START = re.compile(rb"\] n:\s*(\d+) pts:\s*(-?\d+)")
_CHECKSUM = re.compile(rb"checksum:([A-Fa-f0-9]{8})")
_TIME_BASE = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")


def showinfo_frames(output: bytes) -> list[tuple[int, int, int]]:
    """Parse frames even when decoder warnings interrupt a showinfo line."""
    starts = list(_FRAME_START.finditer(output))
    frames = []
    for position, match in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(output)
        checksum = _CHECKSUM.search(output, match.end(), end)
        if checksum:
            frames.append((int(match.group(1)), int(match.group(2)),
                           int(checksum.group(1), 16)))
    return frames


def index_path(video_id: str, index, directory: Path = PTS_DIR, *, decoder_threads: int | None = None) -> Path:
    source = index.zip_path.stat()
    parts = [
        str(index.zip_path.resolve()), source.st_size, source.st_mtime_ns,
        index.data_offset, index.video_size,
    ]
    # N frame numbers are counted across decoder/filter reinitializations.
    # Older maps used showinfo's local n and a different FFmpeg binary.
    if video_id.startswith("N"):
        parts.append("system-ffmpeg-global-frame-v2")
        threads = source_map_decoder_threads(video_id, index) if decoder_threads is None else decoder_threads
        if threads != 4:
            parts.append(f"decoder-threads={threads}")
    identity = json.dumps(parts)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return directory / f"{video_id}-{digest}.npy"


def read_exact_pts(video_id: str, index, frame_number: int,
                   directory: Path = PTS_DIR) -> tuple[int, int, int | None, int | None, int]:
    """Return decoded PTS context and the requested frame's pixel checksum."""
    path = index_path(video_id, index, directory)
    try:
        source = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"Exact frame index missing for {video_id}: {path}")
    identity = (source.st_mtime_ns, source.st_size)
    if (getattr(index, "exact_frame_pts", None) is None or
            getattr(index, "exact_frame_pts_identity", None) != identity):
        table = np.load(path, mmap_mode="r", allow_pickle=False)
        if table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64 or len(table) == 0:
            raise ValueError(f"Invalid exact frame index: {path}")
        if table[0, 0] != 0 or np.any(table[1:, 0] <= table[:-1, 0]):
            raise ValueError(f"Unordered exact frame index: {path}")
        index.exact_frame_pts = table
        index.exact_frame_pts_identity = identity
    table = index.exact_frame_pts
    position = int(np.searchsorted(table[:, 0], frame_number))
    if position >= len(table) or int(table[position, 0]) != frame_number:
        raise ValueError(f"Frame {frame_number} is absent from the exact index for {video_id}")
    previous = int(table[position - 1, 1]) if position and table[position - 1, 0] == frame_number - 1 else None
    following = (int(table[position + 1, 1]) if position + 1 < len(table)
                 and table[position + 1, 0] == frame_number + 1 else None)
    return int(table[0, 1]), int(table[position, 1]), previous, following, int(table[position, 2])


def read_full_timeline_us(video_id: str, index, directory: Path = PTS_DIR) -> bytes:
    """Return the complete N-video presentation clock as little-endian u32 microseconds."""
    path = index_path(video_id, index, directory)
    table = np.load(path, mmap_mode="r", allow_pickle=False)
    if (table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64
            or len(table) < 2 or not np.array_equal(table[:, 0], np.arange(len(table)))):
        raise ValueError(f"Full frame timeline is unavailable for {video_id}")
    pts = table[:, 1]
    if np.any(pts[1:] < pts[:-1]):
        raise ValueError(f"Nonmonotonic frame timeline for {video_id}")
    relative_us = np.floor_divide((pts - pts[0]) * 1_000_000, index.timescale)
    if relative_us[-1] >= 2**32:
        raise ValueError(f"Frame timeline exceeds u32 range for {video_id}")
    return relative_us.astype("<u4", copy=False).tobytes()


def build_exact_pts(video_id: str, index, selected_frames: list[int],
                    frame_count: int, ffmpeg_bin: str, directory: Path = PTS_DIR,
                    *, force_dense: bool = False,
                    store_full_timeline: bool = False,
                    decoder_threads: int | None = None) -> Path:
    """Decode once and record PTS for selected output indices, without images."""
    selected = {int(frame) for frame in selected_frames}
    if frame_count < 1 or any(frame < 0 or frame >= frame_count for frame in selected):
        raise ValueError(f"Out-of-range frame number for {video_id}")
    targets = (None if store_full_timeline else
               sorted({0, *(neighbor for frame in selected for neighbor in
                           (frame - 1, frame, frame + 1) if 0 <= neighbor < frame_count)}))
    threads = source_map_decoder_threads(video_id, index) if decoder_threads is None else decoder_threads
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError('Decoder threads must be an integer in 1..16')
    path = index_path(video_id, index, directory, decoder_threads=threads)
    if path.is_file():
        try:
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            valid_full = (store_full_timeline and existing.ndim == 2 and
                          existing.shape[1] == 3 and len(existing) == frame_count
                          and np.array_equal(existing[:, 0], np.arange(len(existing))))
            valid_sparse = (not store_full_timeline and existing.shape == (len(targets), 3)
                            and np.array_equal(existing[:, 0], targets))
            if existing.dtype == np.int64 and (valid_full or valid_sparse):
                return path
        except (OSError, ValueError):
            pass

    source = (
        f"subfile,,start,{index.data_offset},end,"
        f"{index.data_offset + index.video_size},,:{index.zip_path}"
    )
    # For the short N-series videos, selecting a few frames avoids writing
    # tens of thousands of showinfo lines. S-series has thousands of selected
    # frames, so one showinfo pass with constant work per decoded frame is faster.
    sparse = not store_full_timeline and len(targets) <= 64 and not force_dense
    select = "+".join(f"eq(n\\,{frame})" for frame in targets) if sparse else ""
    vf = f"select='{select}',showinfo" if sparse else "showinfo"
    command = [
        ffmpeg_bin, "-hide_banner", "-nostats", "-loglevel", "info", "-copyts",
        "-threads", str(threads),
        "-i", source, "-map", "0:v:0", "-an", "-vf", vf,
        "-filter_threads", "1",
        "-vsync", "0",
    ]
    if not store_full_timeline:
        command += ["-frames:v", str(len(targets) if sparse else targets[-1] + 1)]
    command += ["-f", "null", "-"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stderr is not None
    found: dict[int, tuple[int, int]] = {}
    decoded_frames = 0
    time_base = None
    target_set = set(targets) if targets is not None else None
    recent_errors: list[str] = []
    unparsed: list[str] = []

    def capture(record: bytes) -> None:
        nonlocal decoded_frames
        starts = list(_FRAME_START.finditer(record))
        parsed = showinfo_frames(record)
        if starts and not parsed:
            unparsed.append(record[:300].decode(errors="replace"))
            del unparsed[:-5]
        for _, pts, checksum in parsed:
            # showinfo's n starts at zero again when FFmpeg rebuilds a filter
            # graph. The preprocessing pipeline counts frames across that
            # boundary, so use the output stream order as the global index.
            output_number = decoded_frames
            decoded_frames += 1
            number = targets[output_number] if sparse and output_number < len(targets) else output_number
            if target_set is None or number in target_set:
                found[number] = (pts, checksum)

    record = bytearray()
    try:
        for line in process.stderr:
            base = _TIME_BASE.search(line)
            if base:
                time_base = (int(base.group(1)), int(base.group(2)))
            start = _FRAME_START.search(line)
            if start:
                record.extend(line[:start.start()])
                capture(record)
                record = bytearray(line[start.start():])
            else:
                record.extend(line)
            if b"Error" in line or b"Invalid" in line:
                recent_errors.append(line.decode(errors="replace")[-300:])
                recent_errors = recent_errors[-5:]
        capture(record)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stderr.close()
    returncode = process.wait()
    if store_full_timeline:
        missing = sorted(set(range(max(found, default=-1) + 1)) - found.keys())[:8]
        missing_selected = sorted(selected - found.keys())[:8]
        complete = (decoded_frames == frame_count and bool(found) and 0 in found
                    and not missing and not missing_selected)
        targets = list(range(max(found) + 1)) if complete else []
    else:
        missing = sorted(target_set - found.keys())[:8]
        complete = len(found) == len(targets)
    if returncode != 0 or not complete:
        raise RuntimeError(
            f"{video_id}: FFmpeg exited {returncode}; decoded {decoded_frames}/{frame_count}; "
            f"exact PTS missing for {missing}; "
            f"selected missing {missing_selected if store_full_timeline else []}; "
            + " ".join(recent_errors + unparsed)
        )
    if time_base != (1, index.timescale):
        raise RuntimeError(
            f"{video_id}: decoded time base {time_base} differs from track 1/{index.timescale}"
        )
    table = np.asarray(
        [(frame, found[frame][0], found[frame][1]) for frame in targets], dtype=np.int64
    )
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as file:
            temporary = file.name
            np.save(file, table)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path
