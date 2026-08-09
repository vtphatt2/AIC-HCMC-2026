"""
WORKAROUND: approximate frame thumbnails via YouTube's storyboard sprites.

This is NOT the real extracted keyframe. It fetches the same low-res sprite
sheet YouTube uses for the scrubber-hover preview and crops out the tile
nearest to the requested timestamp. Accuracy is limited by the storyboard's
own tile interval (commonly a few seconds apart, coarser on long videos) — it
is meant to give SAMPLE-mode/local testing something visual to look at when
the real dataset keyframes aren't present, not to stand in for production
frame images.

See docs/youtube-storyboard-thumbnails-workaround.md for configuration,
accuracy caveats, and when to use (or not use) this.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class StoryboardUnavailable(RuntimeError):
    """Raised when a storyboard thumbnail cannot be produced for a video/timestamp."""


def _find_repo_root() -> Path:
    """Walk up from this file looking for the challenge_resources/ marker
    directory (same trick used in text_encoder.py — remote-server and
    local-backend nest this file at different depths, so we don't hardcode a
    parents[] index)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "challenge_resources").is_dir():
            return parent
    return here.parents[-1]


def _cache_dir() -> Path:
    override = os.getenv("YOUTUBE_STORYBOARD_CACHE_DIR")
    path = Path(override) if override else _find_repo_root() / "cache" / "youtube_storyboards"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _StoryboardIndex:
    """Parsed storyboard metadata for one YouTube video (no image bytes yet)."""

    def __init__(self, youtube_id: str, fmt: dict):
        self.youtube_id = youtube_id
        self.width = int(fmt["width"])
        self.height = int(fmt["height"])
        self.rows = int(fmt["rows"])
        self.columns = int(fmt["columns"])
        self.fps = float(fmt["fps"])  # storyboard tiles per second (i.e. 1 tile every 1/fps seconds)
        self.fragments = fmt["fragments"]  # [{"url": ..., "duration": seconds}, ...]

    def tile_for_timestamp(self, timestamp_ms: int) -> tuple[str, int, int]:
        """Returns (fragment_url, row, col) for the tile nearest this timestamp."""
        seconds = max(0.0, timestamp_ms / 1000.0)
        tiles_per_fragment = self.rows * self.columns
        global_tile_index = int(seconds * self.fps)

        fragment_index = global_tile_index // tiles_per_fragment
        fragment_index = min(fragment_index, len(self.fragments) - 1)
        tile_offset = global_tile_index - fragment_index * tiles_per_fragment
        tile_offset = min(tile_offset, tiles_per_fragment - 1)

        row = tile_offset // self.columns
        col = tile_offset % self.columns
        return self.fragments[fragment_index]["url"], row, col


_index_lock = threading.Lock()


@lru_cache(maxsize=256)
def _get_storyboard_index(youtube_id: str) -> _StoryboardIndex:
    try:
        import yt_dlp
    except ImportError as exc:
        raise StoryboardUnavailable(
            "yt-dlp is not installed. Install it with: python -m pip install yt-dlp pillow"
        ) from exc

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=False)
    except Exception as exc:
        raise StoryboardUnavailable(f"Could not fetch storyboard info for youtube_id={youtube_id}: {exc}") from exc

    storyboard_formats = [f for f in info.get("formats", []) if f.get("format_note") == "storyboard" and f.get("fragments")]
    if not storyboard_formats:
        raise StoryboardUnavailable(f"No storyboard formats found for youtube_id={youtube_id}")

    # Highest fps = finest tile interval = closest match to the requested timestamp.
    best = max(storyboard_formats, key=lambda f: f.get("fps") or 0)
    return _StoryboardIndex(youtube_id, best)


def prefetch_storyboard_index(youtube_id: str) -> None:
    """Warms the per-video storyboard-index cache ahead of the first preview
    request — called from the search endpoint alongside the precise-frame
    prefetch, so the (also yt-dlp-backed, ~2-3s) storyboard lookup mostly
    finishes before the frontend requests a placeholder thumbnail. Best
    effort: swallows failures."""
    try:
        with _index_lock:
            _get_storyboard_index(youtube_id)
    except StoryboardUnavailable:
        logger.debug("Storyboard prefetch failed for youtube_id=%s", youtube_id, exc_info=True)


def _fetch_fragment_bytes(url: str) -> bytes:
    import hashlib

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    cache_path = _cache_dir() / f"{digest}.jpg"
    if cache_path.is_file():
        return cache_path.read_bytes()

    import httpx

    with httpx.Client(timeout=15.0) as client:
        response = client.get(url)
        response.raise_for_status()
        content = response.content

    cache_path.write_bytes(content)
    return content


def get_thumbnail_jpeg(youtube_id: str, timestamp_ms: int) -> bytes:
    """Returns a cropped JPEG tile approximating the frame at timestamp_ms."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise StoryboardUnavailable(
            "Pillow is not installed. Install it with: python -m pip install yt-dlp pillow"
        ) from exc

    with _index_lock:
        index = _get_storyboard_index(youtube_id)

    fragment_url, row, col = index.tile_for_timestamp(timestamp_ms)
    sprite_bytes = _fetch_fragment_bytes(fragment_url)

    import io

    sprite = Image.open(io.BytesIO(sprite_bytes))
    box = (col * index.width, row * index.height, (col + 1) * index.width, (row + 1) * index.height)
    tile = sprite.crop(box)

    buffer = io.BytesIO()
    tile.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
