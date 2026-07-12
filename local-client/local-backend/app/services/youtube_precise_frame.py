"""
Precise frame extraction: resolve the real YouTube video stream via yt-dlp,
then seek into it with ffmpeg and decode exactly one frame at the requested
timestamp. Sharper and timestamp-accurate compared to the storyboard-crop
workaround in youtube_thumbnail.py, at the cost of being slower per frame on
a cache miss (network seek + decode instead of a cheap sprite crop).

ffmpeg comes from the `imageio-ffmpeg` package (bundles a static binary), so
no system-wide ffmpeg install is required.

See docs/youtube-storyboard-thumbnails-workaround.md for background on why
this whole family of workarounds exists (no real extracted keyframes present
in SAMPLE mode).
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class PreciseFrameUnavailable(RuntimeError):
    """Raised when a precise frame cannot be extracted for a video/timestamp."""


def _find_repo_root() -> Path:
    """Same trick used in youtube_thumbnail.py / text_encoder.py — remote-server
    and local-backend nest this file at different depths, so don't hardcode
    a parents[] index."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "onnx-models").is_dir():
            return parent
    return here.parents[-1]


def _cache_dir() -> Path:
    override = os.getenv("YOUTUBE_PRECISE_CACHE_DIR")
    path = Path(override) if override else _find_repo_root() / "cache" / "youtube_precise_frames"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Disk cache has no expiry (every distinct frame ever viewed is kept), so cap
# its total size and evict the least-recently-used files once over budget.
_MAX_CACHE_BYTES = int(os.getenv("YOUTUBE_PRECISE_CACHE_MAX_MB", "300")) * 1024 * 1024
_cache_cleanup_lock = threading.Lock()


def _enforce_cache_limit(cache_dir: Path) -> None:
    if not _cache_cleanup_lock.acquire(blocking=False):
        return  # a cleanup is already running elsewhere; skip rather than block a request
    try:
        files = list(cache_dir.glob("*.jpg"))
        entries = []
        total = 0
        for f in files:
            try:
                stat = f.stat()
            except OSError:
                continue
            # mtime, not atime: Windows disables last-access-time updates by
            # default, so atime can't be trusted for LRU here. get_precise_frame_jpeg
            # touches mtime on every cache hit to keep this meaningful.
            entries.append((f, stat.st_mtime, stat.st_size))
            total += stat.st_size

        if total <= _MAX_CACHE_BYTES:
            return

        entries.sort(key=lambda e: e[1])  # oldest-accessed first
        for f, _atime, size in entries:
            if total <= _MAX_CACHE_BYTES:
                break
            try:
                f.unlink()
                total -= size
            except OSError:
                pass
    finally:
        _cache_cleanup_lock.release()


# Direct googlevideo.com stream URLs expire a few hours after yt-dlp resolves
# them; re-resolve well before that so a long dev session doesn't hit stale
# links. Cached per youtube_id so a whole results grid (mostly a handful of
# distinct videos) only pays the yt-dlp network round-trip once per video.
_URL_TTL_SECONDS = 4 * 60 * 60
_url_cache: dict[str, tuple[str, float]] = {}
_url_cache_lock = threading.Lock()


def _resolve_stream_url(youtube_id: str) -> str:
    with _url_cache_lock:
        cached = _url_cache.get(youtube_id)
        if cached and (time.monotonic() - cached[1]) < _URL_TTL_SECONDS:
            return cached[0]

    try:
        import yt_dlp
    except ImportError as exc:
        raise PreciseFrameUnavailable(
            "yt-dlp is not installed. Install it with: python -m pip install yt-dlp imageio-ffmpeg"
        ) from exc

    # Modest resolution — sharp enough for a results-grid thumbnail, small
    # enough that ffmpeg's seek+decode stays fast. Tunable since it's a
    # direct speed/sharpness trade-off with no one right answer.
    max_height = os.getenv("YOUTUBE_PRECISE_MAX_HEIGHT", "360").strip()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": f"bestvideo[height<={max_height}][ext=mp4]/best[height<={max_height}]/worst",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=False)
    except Exception as exc:
        raise PreciseFrameUnavailable(f"Could not resolve stream for youtube_id={youtube_id}: {exc}") from exc

    url = info.get("url")
    if not url:
        # yt-dlp didn't collapse to a single selected format — pick one
        # ourselves from the full formats list.
        formats = info.get("formats") or []
        candidates = [f for f in formats if f.get("url") and f.get("vcodec") not in (None, "none")]
        if not candidates:
            raise PreciseFrameUnavailable(f"No playable video stream for youtube_id={youtube_id}")
        candidates.sort(key=lambda f: f.get("height") or 0)
        max_h = int(max_height) if max_height.isdigit() else 360
        chosen = next((f for f in candidates if (f.get("height") or 0) <= max_h), candidates[0])
        url = chosen["url"]

    with _url_cache_lock:
        _url_cache[youtube_id] = (url, time.monotonic())
    return url


def prefetch_stream_url(youtube_id: str) -> None:
    """Warms the resolved-URL cache for youtube_id ahead of the first frame
    request — called from the search endpoint right after results are built,
    so the slow yt-dlp lookup mostly happens while the frontend is still
    rendering the page instead of blocking the first thumbnail load. Best
    effort: swallows failures since this is a background optimization, not
    something a request should fail over."""
    try:
        _resolve_stream_url(youtube_id)
    except PreciseFrameUnavailable:
        logger.debug("Prefetch failed for youtube_id=%s", youtube_id, exc_info=True)


def _ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise PreciseFrameUnavailable(
            "imageio-ffmpeg is not installed. Install it with: python -m pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_precise_frame_jpeg(youtube_id: str, timestamp_ms: int, timeout_sec: float = 8.0) -> bytes:
    """Extracts the exact frame at timestamp_ms straight from the YouTube
    video stream via ffmpeg. Cached on disk per (youtube_id, timestamp) so
    repeat requests for the same frame are instant after the first miss."""
    seconds = max(0.0, timestamp_ms / 1000.0)
    cache_key = f"{youtube_id}_{round(seconds * 2) / 2}"  # snap to nearest 0.5s
    cache_path = _cache_dir() / f"{cache_key}.jpg"
    if cache_path.is_file():
        try:
            os.utime(cache_path, None)  # mark as recently used for LRU eviction
        except OSError:
            pass
        return cache_path.read_bytes()

    stream_url = _resolve_stream_url(youtube_id)
    ffmpeg = _ffmpeg_path()

    # -ss before -i is a fast container-level seek (not a full decode from
    # the start); -frames:v 1 stops ffmpeg after the first decoded frame.
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{seconds:.3f}",
        "-i", stream_url,
        "-frames:v", "1",
        "-q:v", "4",
        "-f", "image2pipe",  # image2 (file-pattern muxer) silently mis-writes when piped to stdout
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreciseFrameUnavailable(f"ffmpeg timed out extracting frame for youtube_id={youtube_id}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace")[:300] if exc.stderr else ""
        raise PreciseFrameUnavailable(f"ffmpeg failed for youtube_id={youtube_id}: {stderr}") from exc

    jpeg_bytes = result.stdout
    if not jpeg_bytes:
        raise PreciseFrameUnavailable(f"ffmpeg produced no output for youtube_id={youtube_id} at {seconds}s")

    cache_path.write_bytes(jpeg_bytes)
    _enforce_cache_limit(cache_path.parent)
    return jpeg_bytes
