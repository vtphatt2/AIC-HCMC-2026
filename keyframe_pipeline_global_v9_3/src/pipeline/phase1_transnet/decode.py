"""Decode raw TransNetV2 input frames from video and pick keyframes from scenes.

Two decode strategies:
  - `decode_transnet_frames` / `_decode_transnet_entry`: one ffmpeg pass, wait for it to finish,
    return the whole frame array. Used by the legacy sequential TransNet phase.
  - `_stream_transnet_entry`: read ffmpeg's stdout incrementally and emit each 100-frame
    TransNetV2 window (25 left-context + 50 output + 25 right-context) as soon as it is fully
    decoded, via `event_q`. Used by the streaming-global TransNet phase so the GPU can start
    after ~75 decoded frames instead of waiting for an entire video.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io_utils import safe_video_id
from ..keyframe_selection import keyframe_count, select_keyframes
from ..video_probe import probe_video
from ..zip_source import VideoEntry, materialized_video

TRANSNET_W = 48
TRANSNET_H = 27
TRANSNET_FRAME_BYTES = TRANSNET_W * TRANSNET_H * 3


@dataclass
class TransNetDecodedVideo:
    entry: VideoEntry
    video_id: str
    fps: float
    width: int
    height: int
    frames: np.ndarray
    decode_seconds: float


@dataclass(frozen=True)
class TransNetDecodeFailure:
    entry: VideoEntry
    video_id: str
    error: str
    detail: str


def decode_transnet_frames(ffmpeg_bin: str, video_source: str) -> np.ndarray:
    cmd = [
        ffmpeg_bin, "-v", "error", "-i", video_source,
        "-map", "0:v:0", "-an",
        "-vf", f"scale={TRANSNET_W}:{TRANSNET_H}:flags=area",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg TransNet decode failed:\n{stderr.decode(errors='replace')}")
    if not stdout or len(stdout) % TRANSNET_FRAME_BYTES:
        raise RuntimeError(f"Invalid TransNet raw byte count: {len(stdout)}")
    count = len(stdout) // TRANSNET_FRAME_BYTES
    return np.frombuffer(stdout, dtype=np.uint8).reshape(count, TRANSNET_H, TRANSNET_W, 3).copy()


def _decode_transnet_entry(args, entry: VideoEntry) -> TransNetDecodedVideo | TransNetDecodeFailure:
    video_id = safe_video_id(entry.name)
    try:
        with tempfile.TemporaryDirectory(prefix=f"transnet-{video_id}-") as temp_name:
            with materialized_video(args.zip_path, entry, Path(temp_name)) as source:
                fps, width, height = probe_video(args.ffprobe_bin, source)
                started = time.perf_counter()
                frames = decode_transnet_frames(args.ffmpeg_bin, source)
                decode_s = time.perf_counter() - started
        if len(frames) == 0:
            raise RuntimeError("decoded zero TransNet frames")
        return TransNetDecodedVideo(entry, video_id, fps, width, height, frames, decode_s)
    except Exception as exc:
        return TransNetDecodeFailure(entry, video_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())


@dataclass(frozen=True)
class TransNetStreamStart:
    entry: VideoEntry
    video_id: str
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class TransNetWindowItem:
    entry: VideoEntry
    video_id: str
    output_start: int
    frames: np.ndarray  # uint8 [100,27,48,3]


@dataclass(frozen=True)
class TransNetStreamDone:
    entry: VideoEntry
    video_id: str
    num_frames: int
    decode_seconds: float


def _window_from_rolling_buffer(
    frame_buffer: deque,
    buffer_start: int,
    first_frame: np.ndarray,
    last_frame: np.ndarray,
    output_start: int,
    total_frames: int | None,
) -> np.ndarray:
    """Build exactly the 25-left + 50-useful + 25-right TransNet window.

    Before EOF, callers only request a window when all required right-context
    frames are already decoded. At EOF total_frames enables right-edge replication.
    """
    out = np.empty((100, TRANSNET_H, TRANSNET_W, 3), dtype=np.uint8)
    for j, global_idx in enumerate(range(output_start - 25, output_start + 75)):
        if global_idx < 0:
            out[j] = first_frame
        elif total_frames is not None and global_idx >= total_frames:
            out[j] = last_frame
        else:
            local_idx = global_idx - buffer_start
            if local_idx < 0 or local_idx >= len(frame_buffer):
                raise RuntimeError(
                    f"rolling buffer miss: global={global_idx} buffer_start={buffer_start} "
                    f"buffer_len={len(frame_buffer)} output_start={output_start} total={total_frames}"
                )
            out[j] = frame_buffer[local_idx]
    return out


def _stream_transnet_entry(args, entry: VideoEntry, event_q) -> None:
    """Decode one video incrementally and emit 100-frame TransNet windows immediately."""
    video_id = safe_video_id(entry.name)
    try:
        with tempfile.TemporaryDirectory(prefix=f"transnet-stream-{video_id}-") as temp_name:
            with materialized_video(args.zip_path, entry, Path(temp_name)) as source:
                fps, width, height = probe_video(args.ffprobe_bin, source)
                event_q.put(TransNetStreamStart(entry, video_id, fps, width, height))
                cmd = [
                    args.ffmpeg_bin, "-v", "error", "-i", source,
                    "-map", "0:v:0", "-an",
                    "-vf", f"scale={TRANSNET_W}:{TRANSNET_H}:flags=area",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=TRANSNET_FRAME_BYTES * 64)
                if proc.stdout is None or proc.stderr is None:
                    raise RuntimeError("failed to open ffmpeg pipes")

                started = time.perf_counter()
                frame_buffer: deque[np.ndarray] = deque()
                buffer_start = 0
                first_frame: np.ndarray | None = None
                last_frame: np.ndarray | None = None
                num_frames = 0
                next_output_start = 0
                carry = bytearray()
                read_size = TRANSNET_FRAME_BYTES * 64

                while True:
                    chunk = proc.stdout.read(read_size)
                    if not chunk:
                        break
                    carry.extend(chunk)
                    complete_bytes = (len(carry) // TRANSNET_FRAME_BYTES) * TRANSNET_FRAME_BYTES
                    if complete_bytes == 0:
                        continue
                    raw = bytes(carry[:complete_bytes])
                    del carry[:complete_bytes]
                    block = np.frombuffer(raw, dtype=np.uint8).reshape(-1, TRANSNET_H, TRANSNET_W, 3)
                    for frame in block:
                        frame_copy = frame.copy()
                        if first_frame is None:
                            first_frame = frame_copy
                        last_frame = frame_copy
                        frame_buffer.append(frame_copy)
                        num_frames += 1

                        # A window for output_start s needs frames through s+74.
                        while first_frame is not None and num_frames >= next_output_start + 75:
                            window = _window_from_rolling_buffer(
                                frame_buffer, buffer_start, first_frame, last_frame,
                                next_output_start, total_frames=None,
                            )
                            event_q.put(TransNetWindowItem(entry, video_id, next_output_start, window))
                            next_output_start += 50
                            keep_from = max(0, next_output_start - 25)
                            while buffer_start < keep_from and frame_buffer:
                                frame_buffer.popleft()
                                buffer_start += 1

                stderr = proc.stderr.read()
                returncode = proc.wait()
                decode_seconds = time.perf_counter() - started
                if returncode != 0:
                    raise RuntimeError(f"ffmpeg TransNet decode failed:\n{stderr.decode(errors='replace')}")
                if carry:
                    raise RuntimeError(f"Invalid TransNet raw byte count: trailing {len(carry)} byte(s)")
                if num_frames == 0 or first_frame is None or last_frame is None:
                    raise RuntimeError("decoded zero TransNet frames")

                # Flush final windows that require right-edge replication.
                while next_output_start < num_frames:
                    window = _window_from_rolling_buffer(
                        frame_buffer, buffer_start, first_frame, last_frame,
                        next_output_start, total_frames=num_frames,
                    )
                    event_q.put(TransNetWindowItem(entry, video_id, next_output_start, window))
                    next_output_start += 50
                    keep_from = max(0, next_output_start - 25)
                    while buffer_start < keep_from and frame_buffer:
                        frame_buffer.popleft()
                        buffer_start += 1

                event_q.put(TransNetStreamDone(entry, video_id, num_frames, decode_seconds))
    except Exception as exc:
        event_q.put(TransNetDecodeFailure(entry, video_id, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
