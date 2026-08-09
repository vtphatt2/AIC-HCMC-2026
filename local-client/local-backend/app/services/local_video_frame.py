"""Extract exact JPEG frames from ``data/videos/<video_id>.mp4`` with ffmpeg."""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path


class LocalFrameUnavailable(RuntimeError):
    pass


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "challenge_resources").is_dir():
            return parent
    raise LocalFrameUnavailable("Repository root could not be located")


def local_video_path(video_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("Invalid video_id")
    data_root = Path(os.getenv("AIC_SAMPLE_ROOT", _repo_root() / "challenge_resources" / "data"))
    path = data_root / "videos" / f"{video_id}.mp4"
    if not path.is_file():
        raise LocalFrameUnavailable(f"Local video not found: {path}")
    return path


def _ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise LocalFrameUnavailable(
            "imageio-ffmpeg is required: python -m pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=512)
def get_local_frame_jpeg(
    video_id: str,
    timestamp_ms: int,
    timeout_sec: float = 8.0,
) -> bytes:
    """Fast-seek into a local MP4 and decode one frame; never accesses YouTube."""
    command = [
        _ffmpeg_path(),
        "-loglevel", "error",
        "-ss", f"{max(0, int(timestamp_ms)) / 1000:.3f}",
        "-i", str(local_video_path(video_id)),
        "-frames:v", "1",
        "-q:v", "4",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalFrameUnavailable(f"ffmpeg timed out for video_id={video_id}") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode(errors="replace")[:300] if exc.stderr else ""
        raise LocalFrameUnavailable(f"ffmpeg failed for video_id={video_id}: {error}") from exc
    if not result.stdout:
        raise LocalFrameUnavailable(f"ffmpeg produced no frame for video_id={video_id}")
    return result.stdout
