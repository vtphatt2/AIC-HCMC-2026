"""Serve frames and video playback from a **local** copy of the organizers'
`Videos_L*.zip` archives, without unpacking them.

Same sample-table design as `local-client/local-backend/app/services/zip_frame_source.py`,
with the fetch layer swapped: that one issues HTTP Range requests against the
organizers' host, this one seeks in a file on this machine. Everything that
module carries to survive the network — retry/backoff, the on-disk `moov`
cache, the GOP region cache, single-flight — is dropped here, because a seek
into a local file costs nothing to repeat. What stays is the parsed sample
index per video, which is CPU work, not I/O. A bounded JPEG cache avoids
repeating ffmpeg work when several viewers request the same immutable frame.

Entries inside the archives are stored uncompressed (`ZIP_STORED`), so byte N of
a video is byte `data_offset + N` of the archive and any range is one seek away.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import struct
import time
import zipfile
from array import array
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services import mp4_box_parser as box
from app.services.exact_frame_pts import index_path, read_exact_pts, read_full_timeline_us, showinfo_frames
from app.services.exact_frame_images import verified_exact_image_bytes
from app.services.jpeg_disk_cache import JpegDiskCache

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_HEADER_SIZE = 30
LOCAL_HEADER_STRUCT = "<4s5H3L2H"
ZIP_STORED = 0
STREAM_CHUNK = 1024 * 1024
# How far past the target to look for frames that are presented before it.
# H.264 reordering depth is small; 32 is far past any real encoder setting.
REORDER_LOOKAHEAD = 32
PRESENTATION_FRAME_CACHE_VERSION = "presentation-v4"
N_FRAME_CACHE_VERSION = "system-ffmpeg-global-frame-v6"

# ffmpeg is the only real cost on this path; reading the bytes is a local seek.
_decode_semaphore = asyncio.Semaphore(int(os.getenv("ZIP_FRAME_DECODE_CONCURRENCY", "6")))
# Resizing a ready JPEG must not wait behind a burst of queued video decodes.
_resize_semaphore = asyncio.Semaphore(2)
FFMPEG_THREADS = max(0, int(os.getenv("ZIP_FFMPEG_THREADS", "0")))
JPEG_CACHE_MAX_BYTES = max(0, int(os.getenv("ZIP_JPEG_CACHE_MB", "0"))) * 1024 * 1024
_jpeg_cache: OrderedDict[tuple[str, int, int | None, str, int | None, str | None], bytes] = OrderedDict()
_jpeg_cache_bytes = 0
_jpeg_inflight: dict[tuple[str, int, int | None, str, int | None, str | None], asyncio.Task[bytes]] = {}
_jpeg_waiters: dict[asyncio.Task[bytes], int] = {}
_disk_cache_dir = os.getenv("ZIP_JPEG_DISK_CACHE_DIR", "").strip()
_disk_cache = JpegDiskCache(
    _disk_cache_dir, max(0, int(os.getenv("ZIP_JPEG_DISK_CACHE_MB", "512"))) * 1024 * 1024
) if _disk_cache_dir else None
_card_disk_cache_dir = os.getenv("ZIP_CARD_DISK_CACHE_DIR", "").strip()
_card_disk_cache = JpegDiskCache(
    _card_disk_cache_dir,
    max(0, int(os.getenv("ZIP_CARD_DISK_CACHE_MB", "32768"))) * 1024 * 1024,
) if _card_disk_cache_dir else None
_card_activity_task: asyncio.Task[None] | None = None


def card_activity_file() -> Path | None:
    """Shared heartbeat so low-priority prewarming yields to live card decodes."""
    cache = _card_disk_cache or _disk_cache
    return cache.directory / ".live-card-decode" if cache is not None else None


def _touch_card_activity(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


async def _record_live_card_activity() -> None:
    path = card_activity_file()
    if path is None:
        return
    while _jpeg_inflight:
        try:
            await asyncio.to_thread(_touch_card_activity, path)
        except OSError:
            logger.warning("Could not signal live card activity", exc_info=True)
            return
        await asyncio.sleep(.25)


async def wait_for_live_card_idle(idle_seconds: float = 2.0) -> None:
    """Pause only background work while a backend is decoding live images."""
    path = card_activity_file()
    if path is None:
        return
    while True:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return
        if age >= idle_seconds:
            return
        await asyncio.sleep(min(.25, max(.01, idle_seconds - age)))


def _resize_jpeg(original: bytes, width: int, format: str = "jpeg") -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(original)) as source:
        if source.width <= width and format == "jpeg":
            return original
        width = min(width, source.width)
        size = (width, max(1, round(source.height * width / source.width)))
        image = source.resize(size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        if format == "webp":
            image.save(output, "WEBP", quality=85, method=4)
        else:
            image.save(output, "JPEG", quality=88, subsampling=2, optimize=True)
        return output.getvalue()


def _map_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame_map_generation(video_id: str, index: "VideoFrameIndex") -> str:
    path = index_path(video_id, index)
    return f'{path.stem}-{_map_sha256(str(path))}'


def _cache_key(index: "VideoFrameIndex", video_id: str, timestamp_ms: int,
               width: int | None, format: str, frame_number: int | None) -> str:
    source = index.zip_path.stat()
    # Keep the original key stable; resized images have a separate version.
    version = "jpeg-v1-q4" if width is None else f"jpeg-card-v1-q88-w{width}"
    if format == "webp":
        version = f"webp-card-v1-q85-w{width}"
    if frame_number is not None:
        frame_version = (N_FRAME_CACHE_VERSION if video_id.startswith("N")
                         else PRESENTATION_FRAME_CACHE_VERSION)
        version += f"-{frame_version}"
        if video_id.startswith("N"):
            version += f"-{_frame_map_generation(video_id, index)}"
    key_parts = [version, str(index.zip_path.resolve()), source.st_size,
                 source.st_mtime_ns, index.data_offset, index.fps, video_id, timestamp_ms]
    if frame_number is not None:
        key_parts.append(frame_number)
    return json.dumps(key_parts)


async def _cached_decode(video_id, timestamp_ms, timeout_sec, width=None, format="jpeg",
                         frame_number=None):
    cache = (_card_disk_cache or _disk_cache) if width is not None else _disk_cache
    key = None
    if cache is not None:
        index = await _get_index(video_id)
        key = _cache_key(index, video_id, timestamp_ms, width, format, frame_number)
        if width is not None:
            try:
                cached = await cache.read(key)
                if cached is not None:
                    return cached
            except OSError:
                logger.warning("JPEG disk cache unavailable; resizing without it", exc_info=True)
                cache = None

    # Resolve the original BEFORE taking a thumbnail shard lock. Otherwise two
    # variants hashing to the same shard could deadlock. Reuse the existing
    # original cache/decoder so both sizes depict exactly the same frame.
    original = await get_frame_jpeg(video_id, timestamp_ms, timeout_sec,
                                    frame_number=frame_number) if width else None

    async def decode():
        if original is not None:
            async with _resize_semaphore:
                return await asyncio.to_thread(_resize_jpeg, original, width, format)
        if frame_number is None:
            return await _decode_frame_jpeg(video_id, timestamp_ms, timeout_sec)
        return await _decode_frame_jpeg(video_id, timestamp_ms, timeout_sec,
                                        frame_number=frame_number)
    if cache is None:
        return await decode()
    try:
        return await cache.get(key, decode)
    except OSError:
        logger.warning("JPEG disk cache unavailable; decoding without it", exc_info=True)
        return await decode()



class LocalZipUnavailable(RuntimeError):
    """No archive holds this video_id, or its MP4 could not be parsed."""


# ── Archive index ─────────────────────────────────────────────────────────────

def raw_zip_dir() -> Path:
    configured = os.getenv("RAW_ZIP_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "challenge_resources" / "data" / "raw_zip_videos"


def _data_offset(zip_path: Path, header_offset: int) -> int:
    """Where an entry's raw bytes begin: past its local file header, whose
    filename/extra lengths can differ from the central directory's."""
    with zip_path.open("rb") as handle:
        handle.seek(header_offset)
        header = handle.read(LOCAL_HEADER_SIZE)
    if len(header) < LOCAL_HEADER_SIZE or header[:4] != b"PK\x03\x04":
        raise LocalZipUnavailable(f"Bad local file header at {header_offset} in {zip_path.name}")
    fields = struct.unpack(LOCAL_HEADER_STRUCT, header)
    return header_offset + LOCAL_HEADER_SIZE + fields[9] + fields[10]


def _scan_archives() -> dict[str, dict]:
    """video_id -> {zip_path, entry_name, data_offset, size} for every archive
    in raw_zip_videos/. Only central-directory bytes are read, so this stays fast even
    though the archives are tens of GB."""
    directory = raw_zip_dir()
    if not directory.is_dir():
        logger.info("local zip media: %s does not exist — routes will 404", directory)
        return {}

    index: dict[str, dict] = {}
    for zip_path in sorted(directory.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                infos = archive.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            logger.warning("local zip media: skipping %s (%s)", zip_path.name, exc)
            continue

        found = 0
        for info in infos:
            if not info.filename.lower().endswith((".mp4", ".mov")):
                continue
            if info.compress_type != ZIP_STORED:
                logger.warning(
                    "local zip media: %s/%s is not ZIP_STORED — cannot serve byte ranges",
                    zip_path.name, info.filename,
                )
                continue
            video_id = Path(info.filename).stem
            if video_id in index:
                continue
            try:
                offset = _data_offset(zip_path, info.header_offset)
            except LocalZipUnavailable as exc:
                logger.warning("local zip media: %s", exc)
                continue
            index[video_id] = {
                "zip_path": zip_path,
                "entry_name": info.filename,
                "data_offset": offset,
                "size": info.file_size,
            }
            found += 1
        logger.info("local zip media: %s -> %s videos", zip_path.name, found)

    logger.info("local zip media: %s videos indexed from %s", len(index), directory)
    return index


_archive_index: dict[str, dict] | None = None
_archive_lock = asyncio.Lock()


async def _index() -> dict[str, dict]:
    global _archive_index
    if _archive_index is None:
        async with _archive_lock:
            if _archive_index is None:
                _archive_index = await asyncio.to_thread(_scan_archives)
    return _archive_index


async def lookup(video_id: str) -> dict:
    entry = (await _index()).get(video_id)
    if entry is None:
        raise LocalZipUnavailable(f"No local archive entry for video_id={video_id}")
    return entry


async def available_video_count() -> int:
    return len(await _index())


async def exact_frame_offset_us(video_id: str, frame_number: int) -> int:
    """Presentation time of a search frame relative to this video's first frame."""
    index = await _get_index(video_id)
    first_pts, pts, _, _, _ = read_exact_pts(video_id, index, frame_number)
    return (pts - first_pts) * 1_000_000 // index.timescale


async def full_frame_timeline_us(video_id: str) -> bytes:
    """Compact timestamp lookup for a video's full decoded frame sequence."""
    index = await _get_index(video_id)
    return read_full_timeline_us(video_id, index)


async def full_frame_timeline_v2(video_id: str) -> dict:
    from app.services.source_timeline import timeline_response
    index = await _get_index(video_id)
    return await asyncio.to_thread(timeline_response, video_id, index)


async def timing_capabilities(video_id: str) -> dict:
    """Small capability summary; picture timing never depends on playback."""
    from app.services.exact_frame_pts import index_path
    from app.services.playback_copies import serving_copy
    from app.services.readiness_policy import submission_unit, BROWSER_COMPATIBLE_N_SOURCES
    result = {'submission_unit': submission_unit(video_id),
              'verified_timing': False, 'timeline_version': None,
              'playback_available': True, 'playback_origin_seconds': None,
              'source_time_base': None}
    if not video_id.startswith('N'):
        return result
    try:
        index = await _get_index(video_id)
        result['verified_timing'] = index_path(video_id, index).is_file()
        result['timeline_version'] = 2 if result['verified_timing'] else None
        result['source_time_base'] = {'num': 1, 'den': index.timescale}
        copy = await asyncio.to_thread(serving_copy, video_id, index)
        result['playback_available'] = copy is not None or video_id in BROWSER_COMPATIBLE_N_SOURCES
        if copy:
            result['playback_origin_seconds'] = copy[1]['source_origin_pts'] / index.timescale
    except (LocalZipUnavailable, OSError, ValueError):
        result['playback_available'] = False
    return result


# ── Authoritative fps ─────────────────────────────────────────────────────────
# The fps a video was *ingested* with, not the one derived from its moov.
# `timestamp_ms = frame_number / fps * 1000` was computed at ingest from
# scenes.json; this route inverts it, so it has to invert with the same number.
# They agree for the 25.0 fps majority, but 91 videos are 29.97/30.0 where a
# derived value can land on the wrong one — and that error grows with the
# timestamp instead of staying bounded. Precomputed by
# remote-server/scripts/export_video_fps.py.

def results_zip_dir() -> Path:
    configured = os.getenv("RESULTS_ZIP_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "challenge_resources" / "data" / "zip_embeddings"


def fps_map_path() -> Path:
    configured = os.getenv("VIDEO_FPS_MAP", "").strip()
    if configured:
        return Path(configured).expanduser()
    return results_zip_dir().parent / "video_fps.json"


def _scan_scenes() -> dict[str, float]:
    """Fallback when video_fps.json is absent: read every scenes.json. ~1s for
    873 videos, once per process — cheap enough not to need its own cache file,
    which is exactly what export_video_fps.py already writes."""
    fps_by_video: dict[str, float] = {}
    directory = results_zip_dir()
    if not directory.is_dir():
        return fps_by_video
    for zip_path in sorted(directory.glob("*_results.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for name in archive.namelist():
                    if not name.endswith("/scenes.json"):
                        continue
                    folder = name.split("/")[-2]
                    if not re.fullmatch(r"videos?__.+", folder):
                        continue
                    video_id = folder.split("__", 1)[1]
                    if video_id in fps_by_video:
                        continue
                    try:
                        fps = float(json.loads(archive.read(name))["fps"])
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
                    if fps > 0:
                        fps_by_video[video_id] = fps
        except (OSError, zipfile.BadZipFile) as exc:
            logger.warning("ingest fps: skipping %s (%s)", zip_path.name, exc)
    return fps_by_video


_fps_map: dict[str, float] | None = None


def _load_fps_map() -> dict[str, float]:
    path = fps_map_path()
    try:
        if path.is_file():
            loaded = {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
            logger.info("ingest fps: %s videos from %s", len(loaded), path.name)
            return loaded
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("ingest fps: %s unreadable (%s) — falling back to the archives", path, exc)
    scanned = _scan_scenes()
    logger.info(
        "ingest fps: %s videos scanned from %s (run scripts/export_video_fps.py to skip this)",
        len(scanned), results_zip_dir(),
    )
    return scanned


def ingest_fps(video_id: str) -> float | None:
    """fps from the ingest archives, or None when this video has no record."""
    global _fps_map
    if _fps_map is None:
        _fps_map = _load_fps_map()
    return _fps_map.get(video_id)


# ── Byte access ───────────────────────────────────────────────────────────────

def _read(zip_path: Path, start: int, size: int) -> bytes:
    if size <= 0:
        return b""
    with zip_path.open("rb") as handle:
        handle.seek(start)
        return handle.read(size)


def _parse_range(range_header: str | None, size: int) -> tuple[int, int]:
    """Single-range `bytes=start-end`; whole file when absent."""
    if not range_header:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        raise LocalZipUnavailable(f"Unsupported Range header: {range_header!r}")
    start_s, end_s = match.groups()
    if start_s == "" and end_s == "":
        raise LocalZipUnavailable("Empty Range header")
    if start_s == "":
        start, end = max(0, size - int(end_s)), size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    end = min(end, size - 1)
    if start < 0 or start > end:
        raise LocalZipUnavailable(f"Invalid range {start}-{end} for size {size}")
    return start, end


async def open_range(video_id: str, range_header: str | None):
    """Response headers plus a byte iterator for a client's Range request.

    Content-Range is expressed against the *MP4*, not the archive, so the
    browser's <video> element seeks exactly as it would against a plain URL.
    """
    if video_id.startswith('N'):
        from app.services.playback_copies import open_range as playback_range, serving_copy
        from app.services.readiness_policy import BROWSER_COMPATIBLE_N_SOURCES
        index = await _get_index(video_id)
        if await asyncio.to_thread(serving_copy, video_id, index):
            return await playback_range(video_id, index, range_header)
        if video_id not in BROWSER_COMPATIBLE_N_SOURCES:
            raise LocalZipUnavailable(f'Validated playback unavailable for {video_id}')
    entry = await lookup(video_id)
    size = int(entry["size"])
    data_offset = int(entry["data_offset"])
    start, end = _parse_range(range_header, size)
    zip_path: Path = entry["zip_path"]

    async def body():
        position = data_offset + start
        remaining = end - start + 1
        handle = await asyncio.to_thread(zip_path.open, "rb")
        try:
            await asyncio.to_thread(handle.seek, position)
            while remaining > 0:
                chunk = await asyncio.to_thread(handle.read, min(STREAM_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": "video/mp4",
    }
    return headers, body()


# ── Frame extraction ──────────────────────────────────────────────────────────

@dataclass
class VideoFrameIndex:
    zip_path: Path
    data_offset: int
    video_size: int
    codec: bytes
    timescale: int
    fps: float
    base_pts: int  # pts of the first *presented* frame — rarely 0 when B-frames reorder
    sample_offsets: array
    sample_sizes: array
    sample_pts: array
    keyframe_samples: list[int]
    keyframe_frames: list[int] = field(default_factory=list)
    sps: list[bytes] = field(default_factory=list)
    pps: list[bytes] = field(default_factory=list)
    nal_length_size: int = 4
    presentation_samples: np.ndarray | None = None
    exact_frame_pts: np.ndarray | None = None

    def nearest_keyframe_sample(self, frame_id: int) -> int:
        i = bisect_right(self.keyframe_frames, frame_id) - 1
        return self.keyframe_samples[max(0, i)]

    def presentation_pts(self, frame_number: int) -> int:
        if self.presentation_samples is None:
            self.presentation_samples = np.argsort(
                np.frombuffer(self.sample_pts, dtype=np.int64), kind="stable"
            )
        if frame_number < 0 or frame_number >= len(self.presentation_samples):
            raise LocalZipUnavailable(f"Frame {frame_number} is outside video")
        return self.sample_pts[int(self.presentation_samples[frame_number])]


_index_cache: dict[str, VideoFrameIndex] = {}
_index_locks: dict[str, asyncio.Lock] = {}


def _find_moov(zip_path: Path, mp4_offset: int, mp4_size: int) -> bytes:
    # Top-level ISO BMFF boxes carry their own length. Walk their headers with
    # tiny seeks, skipping mdat without reading it. A moov can exceed either
    # fixed head/tail window (M08 fast-start and S01 tail moov do).
    position = 0
    while position + 8 <= mp4_size:
        header = _read(zip_path, mp4_offset + position, 16)
        if len(header) < 8:
            break
        size = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        header_size = 8
        if size == 1:
            if len(header) < 16:
                break
            size = int.from_bytes(header[8:16], "big")
            header_size = 16
        elif size == 0:
            size = mp4_size - position
        if size < header_size or size > mp4_size - position:
            break
        if kind == b"moov":
            moov = _read(zip_path, mp4_offset + position, size)
            if len(moov) == size:
                return moov
            break
        position += size
    raise LocalZipUnavailable(f"Could not locate a complete moov box in {zip_path.name}")


def _build_index(video_id: str, entry: dict) -> VideoFrameIndex:
    zip_path: Path = entry["zip_path"]
    data_offset = int(entry["data_offset"])
    moov = _find_moov(zip_path, data_offset, int(entry["size"]))

    mdhd, stbl = box.find_video_track(moov)
    timescale, duration = box.parse_mdhd(moov, mdhd)

    stts_box = box.find_child(moov, stbl, b"stts")
    ctts_box = box.find_child(moov, stbl, b"ctts")
    stss_box = box.find_child(moov, stbl, b"stss")
    stsz_box = box.find_child(moov, stbl, b"stsz")
    stsc_box = box.find_child(moov, stbl, b"stsc")
    stco_box = box.find_child(moov, stbl, b"stco") or box.find_child(moov, stbl, b"co64")
    if not all([stts_box, stsz_box, stsc_box, stco_box]):
        raise LocalZipUnavailable(f"{video_id}: required MP4 sample-table box missing")

    sizes = box.parse_stsz(moov, stsz_box)
    sample_count = len(sizes)
    stts = box.parse_stts(moov, stts_box)
    ctts = box.parse_ctts(moov, ctts_box) if ctts_box else []
    key_samples_1based = box.parse_stss(moov, stss_box) if stss_box else [1]
    stsc = box.parse_stsc(moov, stsc_box)
    chunk_offsets = box.parse_chunk_offsets(moov, stco_box)
    offsets = box.build_sample_offsets(sizes, chunk_offsets, stsc)

    dts = box.expand_stts(stts, sample_count)
    cto = box.expand_ctts(ctts, sample_count)
    pts = [dts[i] + cto[i] for i in range(sample_count)]

    avg_delta = duration / sample_count if sample_count else 0
    fps = (timescale / avg_delta) if avg_delta else 0.0
    authoritative = ingest_fps(video_id)
    if authoritative is not None:
        if abs(authoritative - fps) > 1e-6:
            logger.info(
                "%s: using ingest fps %.6f over moov-derived %.6f",
                video_id, authoritative, fps,
            )
        fps = authoritative
    base_pts = min(pts) if pts else 0
    keyset = sorted(s - 1 for s in key_samples_1based)  # 0-based
    keyframe_frames = [
        int(round(((pts[s] - base_pts) / timescale) * fps)) if fps else s
        for s in keyset
    ]
    stsd = box.find_child(moov, stbl, b"stsd")
    if stsd is None:
        raise LocalZipUnavailable(f"{video_id}: missing video sample description")
    sample_entry = stsd["payload_start"] + 8
    codec = moov[sample_entry + 4:sample_entry + 8]
    if codec in (b"avc1", b"avc3"):
        sps, pps, nal_length_size = box.parse_avcc(moov, stbl)
    else:
        sps, pps, nal_length_size = [], [], 4

    return VideoFrameIndex(
        zip_path=zip_path,
        data_offset=data_offset,
        video_size=int(entry["size"]),
        codec=codec,
        timescale=timescale,
        fps=fps,
        base_pts=base_pts,
        # Numeric columns avoid a dictionary and three boxed integers per frame.
        # Keep timestamps signed: composition offsets may precede decode time.
        sample_offsets=array("Q", offsets),
        sample_sizes=array("Q", sizes),
        sample_pts=array("q", pts),
        keyframe_samples=keyset,
        keyframe_frames=keyframe_frames,
        sps=sps,
        pps=pps,
        nal_length_size=nal_length_size,
    )


async def _get_index(video_id: str) -> VideoFrameIndex:
    cached = _index_cache.get(video_id)
    if cached is not None:
        return cached
    entry = await lookup(video_id)
    lock = _index_locks.setdefault(video_id, asyncio.Lock())
    async with lock:
        cached = _index_cache.get(video_id)
        if cached is not None:
            return cached
        index = await asyncio.to_thread(_build_index, video_id, entry)
        _index_cache[video_id] = index
        return index


def _range_plan(index: VideoFrameIndex, frame_id: int, reorder_margin: int = 4):
    """(start_sample, target_sample, end_sample, target_index).

    Samples sit in the file in *decode* order; a decoder emits them in
    *presentation* order, and with B-frames those differ. So the target is
    picked by PTS, and `target_index` is its rank in presentation order within
    the decoded region — which is what ffmpeg's `select=eq(n,…)` counts.
    Using the decode-order position instead returns a neighbouring picture
    (measured: 3 frames off at 60 s on L30_V001).
    """
    # base_pts, not 0: the first presented frame's pts is one or more frame
    # durations in whenever B-frames reorder, and ignoring that returns the
    # frame before the one asked for (measured: consistently -1 on L30_V001).
    target_pts = (
        index.base_pts + (frame_id / index.fps) * index.timescale if index.fps else 0
    )
    start_sample = index.nearest_keyframe_sample(frame_id)
    pts = index.sample_pts
    # Match the scalar int-minus-float distance and its earliest-index tie break.
    # Do not binary-search PTS: B-frames can put them out of presentation order.
    if index.fps:
        distances = np.subtract(
            np.frombuffer(pts, dtype=np.int64)[start_sample:], target_pts, dtype=np.float64,
        )
        np.abs(distances, out=distances)
        target_sample = start_sample + int(distances.argmin())
    else:
        # The no-fps fallback compares integers, including values above 2**53.
        target_sample = min(range(start_sample, len(pts)), key=lambda i: (abs(pts[i]), i))
    target_p = pts[target_sample]

    # Every frame presented before the target must be decoded as well, or the
    # rank below counts a picture the decoder never emitted. Reordering depth is
    # small and bounded, so a fixed look-ahead covers it without scanning on.
    horizon = min(len(pts), target_sample + 1 + REORDER_LOOKAHEAD)
    last_needed = max(
        i for i in range(start_sample, horizon) if pts[i] <= target_p
    )
    end_sample = min(last_needed + reorder_margin, len(pts) - 1)
    target_index = sum(
        1 for i in range(start_sample, end_sample + 1) if pts[i] < target_p
    )
    return start_sample, target_sample, end_sample, target_index


def _avcc_to_annexb(sample_bytes: bytes, nal_length_size: int) -> bytes:
    out = bytearray()
    i, n = 0, len(sample_bytes)
    while i + nal_length_size <= n:
        length = int.from_bytes(sample_bytes[i:i + nal_length_size], "big")
        i += nal_length_size
        out += b"\x00\x00\x00\x01" + sample_bytes[i:i + length]
        i += length
    return bytes(out)


def _ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise LocalZipUnavailable(
            "imageio-ffmpeg is required: python -m pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


async def _run_ffmpeg(command: list[str], timeout_sec: float,
                      input_bytes: bytes | None = None, *,
                      return_stderr: bool = False, allow_empty: bool = False):
    """Run the unchanged FFmpeg command while allowing abandoned work to stop."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, error = await asyncio.wait_for(process.communicate(input_bytes), timeout_sec)
    except BaseException as exc:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        if isinstance(exc, asyncio.TimeoutError):
            raise LocalZipUnavailable("ffmpeg timed out decoding frame") from exc
        raise
    if process.returncode != 0:
        detail = error.decode(errors="replace")[:300] if error else ""
        raise LocalZipUnavailable(f"ffmpeg failed to decode frame: {detail}")
    if not output and not allow_empty:
        raise LocalZipUnavailable("ffmpeg produced no frame")
    return (output, error) if return_stderr else output


async def _decode_jpeg(annexb_stream: bytes, target_index: int, timeout_sec: float) -> bytes:
    command = [
        _ffmpeg_path(),
        "-loglevel", "error",
        "-f", "h264",
        "-threads", str(FFMPEG_THREADS),
        "-i", "pipe:0",
        "-vf", f"select=eq(n\\,{target_index})",
        "-vsync", "0",
        "-frames:v", "1",
        "-q:v", "4",
        "-f", "image2pipe",
        "-threads", "1",
        "-filter_threads", "1",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    return await _run_ffmpeg(command, timeout_sec, annexb_stream)


async def _decode_seekable_jpeg(index: VideoFrameIndex, frame_number: int,
                                timeout_sec: float, video_id: str | None = None) -> bytes:
    """Decode an exact presentation frame from its seekable ZIP_STORED entry.

    FFmpeg's subfile protocol views the entry as the original video without
    copying it. Verify indexed search frames by their decoded pixel checksum:
    a few edited MOVs shift timestamps by several pictures after a seek.
    """
    entry_end = index.data_offset + index.video_size
    source = (
        f"subfile,,start,{index.data_offset},end,{entry_end},,:{index.zip_path}"
    )
    # N timelines are built with the same FFmpeg version used by preprocessing.
    # Its decoded-frame order and showinfo checksums must agree with the map.
    ffmpeg_bin = "/usr/bin/ffmpeg" if video_id and video_id.startswith("N") else _ffmpeg_path()
    if video_id and video_id.startswith("N"):
        from app.services.readiness_policy import verified_embed_decoder_threads
        decoder_threads = verified_embed_decoder_threads(video_id, index)
    else:
        decoder_threads = FFMPEG_THREADS

    async def full_decode(expected_checksum: int | None = None) -> bytes:
        # An arbitrary frame_number may not be in the compact search-frame map.
        # Decoding by n is slower but exact and keeps the public route usable.
        command = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "info", "-copyts",
            "-threads", str(decoder_threads),
            "-i", source, "-map", "0:v:0", "-an",
            "-vf", f"select='eq(n\\,{frame_number})',showinfo", "-vsync", "0",
            "-filter_threads", "1", "-frames:v", "1", "-q:v", "4",
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        image, diagnostics = await _run_ffmpeg(command, timeout_sec, return_stderr=True)
        if expected_checksum is not None and not any(
            checksum == expected_checksum for _, _, checksum in showinfo_frames(diagnostics)
        ):
            raise LocalZipUnavailable(
                f"Decoded frame verification failed for {video_id}/{frame_number}"
            )
        return image

    if video_id is not None:
        try:
            first_pts, pts, previous_pts, next_pts, expected_checksum = read_exact_pts(
                video_id, index, frame_number
            )
        except (FileNotFoundError, ValueError):
            if video_id.startswith("N"):
                raise LocalZipUnavailable(f"Exact frame map is not ready for {video_id}")
            return await full_decode()
        # Search indices refer to decoded output frames, which can diverge from
        # MOV sample indices. A seek can shift a frame PTS by a few track ticks;
        # midpoints to adjacent decoded frames make the common path fast.
        tolerance = max(1, round(index.timescale / index.fps / 4))
        lower = ((previous_pts + pts) // 2 + 1 if previous_pts is not None
                 and previous_pts < pts else pts - tolerance)
        upper = ((pts + next_pts) // 2 if next_pts is not None
                 and next_pts > pts else pts + tolerance)
        seconds = max(0.0, (pts - first_pts) / index.timescale - 0.2)
    else:
        pts = index.presentation_pts(frame_number)
        seconds = max(0.0, (pts - index.base_pts) / index.timescale - 0.001)
    if video_id is None:
        command = [
            _ffmpeg_path(), "-loglevel", "error", "-threads", str(FFMPEG_THREADS),
            "-ss", f"{seconds:.9f}", "-i", source,
            "-map", "0:v:0", "-frames:v", "1", "-q:v", "4",
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        return await _run_ffmpeg(command, timeout_sec)

    def start(seek_seconds: float) -> list[str]:
        return [
            ffmpeg_bin, "-hide_banner", "-loglevel", "info",
            "-threads", str(decoder_threads), "-copyts",
            "-ss", f"{seek_seconds:.9f}", "-i", source, "-map", "0:v:0", "-an",
        ]

    async def one_frame(seek_seconds: float, selection: str) -> tuple[bytes, bytes]:
        command = start(seek_seconds) + [
            "-vf", f"select='{selection}',showinfo", "-vsync", "0",
            "-filter_threads", "1",
            "-frames:v", "1", "-q:v", "4", "-f", "image2pipe",
            "-vcodec", "mjpeg", "pipe:1",
        ]
        return await _run_ffmpeg(command, timeout_sec, return_stderr=True)

    try:
        image, diagnostics = await one_frame(
            seconds, f"between(pts\\,{lower}\\,{upper})"
        )
        if any(checksum == expected_checksum for _, _, checksum in showinfo_frames(diagnostics)):
            return image
    except LocalZipUnavailable:
        pass

    # Locate the correct picture in a short seek window when timestamps move.
    probe_seconds = max(0.0, (pts - first_pts) / index.timescale - 1.0)
    probe = start(probe_seconds) + [
        "-vf", "showinfo", "-vsync", "0", "-filter_threads", "1",
        "-frames:v", str(max(80, round(index.fps * 3))),
        "-f", "null", "-",
    ]
    _, diagnostics = await _run_ffmpeg(
        probe, timeout_sec, return_stderr=True, allow_empty=True
    )
    matches = [seek_pts for _, seek_pts, checksum in showinfo_frames(diagnostics)
               if checksum == expected_checksum]
    if matches:
        sought_pts = min(matches, key=lambda candidate: abs(candidate - pts))
        try:
            image, diagnostics = await one_frame(probe_seconds, f"eq(pts\\,{sought_pts})")
            if any(checksum == expected_checksum for _, _, checksum in showinfo_frames(diagnostics)):
                return image
        except LocalZipUnavailable:
            pass

    if video_id.startswith("N"):
        # A full decode is an offline cache-building task. Repeating it for
        # every live card can exhaust decode slots, and select(n) can reset
        # partway through some N sources. Never return an unverified picture.
        raise LocalZipUnavailable(f"Exact frame needs sequential prewarming: {video_id}/{frame_number}")
    # Preserve the existing fallback for other sources.
    return await full_decode(expected_checksum)


async def get_frame_jpeg(video_id: str, timestamp_ms: int, timeout_sec: float = 20.0,
                         *, width: int | None = None, format: str = "jpeg",
                         frame_number: int | None = None) -> bytes:
    """Share in-flight decodes and retain JPEGs within a per-process byte budget.

    Archive contents are immutable, matching the HTTP cache contract. Shielding
    the shared task lets another viewer finish even when the first disconnects.
    Only successful results are retained; failures can be retried immediately.
    """
    if width not in (None, 640):
        raise ValueError("Only the 640-pixel card variant is supported")
    if format not in ("jpeg", "webp") or (format == "webp" and width is None):
        raise ValueError("WebP is supported only for card previews")
    global _card_activity_task
    map_generation = None
    if frame_number is not None and video_id.startswith('N'):
        try:
            map_generation = _frame_map_generation(video_id, await _get_index(video_id))
        except (OSError, ValueError) as exc:
            raise LocalZipUnavailable(f'Verified source map unavailable for {video_id}') from exc
    key = (video_id, max(0, int(timestamp_ms)), width, format, frame_number, map_generation)
    if key in _jpeg_cache:
        _jpeg_cache.move_to_end(key)
        return _jpeg_cache[key]
    task = _jpeg_inflight.get(key)
    if task is None:
        async def produce():
            global _jpeg_cache_bytes
            try:
                data = await _cached_decode(key[0], key[1], timeout_sec, width=width,
                                            format=format, frame_number=frame_number)
                if len(data) <= JPEG_CACHE_MAX_BYTES:
                    while _jpeg_cache and _jpeg_cache_bytes + len(data) > JPEG_CACHE_MAX_BYTES:
                        _, old = _jpeg_cache.popitem(last=False)
                        _jpeg_cache_bytes -= len(old)
                    _jpeg_cache[key] = data
                    _jpeg_cache_bytes += len(data)
                return data
            finally:
                if _jpeg_inflight.get(key) is asyncio.current_task():
                    _jpeg_inflight.pop(key, None)

        task = asyncio.create_task(produce())
        _jpeg_inflight[key] = task
        if _card_activity_task is None or _card_activity_task.done():
            _card_activity_task = asyncio.create_task(_record_live_card_activity())
        # Retrieve exceptions even if every viewer has disconnected.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    _jpeg_waiters[task] = _jpeg_waiters.get(task, 0) + 1
    try:
        return await asyncio.shield(task)
    finally:
        remaining = _jpeg_waiters[task] - 1
        if remaining:
            _jpeg_waiters[task] = remaining
        else:
            del _jpeg_waiters[task]
            # If every viewer has left, do not spend a decode slot on a frame
            # from an abandoned search. A later viewer starts a fresh task.
            if not task.done():
                if _jpeg_inflight.get(key) is task:
                    _jpeg_inflight.pop(key, None)
                task.cancel()


async def _decode_frame_jpeg(video_id: str, timestamp_ms: int, timeout_sec: float = 20.0,
                             frame_number: int | None = None) -> bytes:
    """JPEG bytes for the frame at `timestamp_ms`, or `LocalZipUnavailable` —
    every failure on this path funnels through that one type so the route never
    has to guess at a raw parse/subprocess error."""
    try:
        index = await _get_index(video_id)

        if frame_number is not None and video_id.startswith("N"):
            # Damaged seek points can produce a different first picture from
            # a clean sequential decode. Verified cached JPEGs are tied to
            # the source-identified timeline; reading one is cheaper and more
            # accurate than repeating a seek that cannot reconstruct it.
            verified = await asyncio.to_thread(verified_exact_image_bytes, video_id, index, frame_number)
            if verified is not None:
                return verified

        if frame_number is not None or index.codec not in (b"avc1", b"avc3"):
            chosen = frame_number if frame_number is not None else (
                round(max(0, int(timestamp_ms)) / 1000 * index.fps) if index.fps else 0
            )
            async with _decode_semaphore:
                return await _decode_seekable_jpeg(
                    index, chosen, timeout_sec, video_id=video_id if frame_number is not None else None
                )

        frame_id = round(max(0, int(timestamp_ms)) / 1000 * index.fps) if index.fps else 0
        start_sample, target_sample, end_sample, target_index = _range_plan(index, frame_id)
        region = range(start_sample, end_sample + 1)
        offsets, sizes = index.sample_offsets, index.sample_sizes
        region_start = min(offsets[i] for i in region)
        region_end = max(offsets[i] + sizes[i] for i in region)

        # Cache final JPEGs above; local reads need no additional region cache.
        fetched = await asyncio.to_thread(
            _read, index.zip_path, index.data_offset + region_start, region_end - region_start,
        )

        stream = bytearray()
        for nal in index.sps:
            stream += b"\x00\x00\x00\x01" + nal
        for nal in index.pps:
            stream += b"\x00\x00\x00\x01" + nal
        for i in region:
            rel = offsets[i] - region_start
            stream += _avcc_to_annexb(fetched[rel:rel + sizes[i]], index.nal_length_size)

        async with _decode_semaphore:
            return await _decode_jpeg(bytes(stream), target_index, timeout_sec)
    except LocalZipUnavailable:
        raise
    except (OSError, RuntimeError, IndexError, ValueError, KeyError, struct.error) as exc:
        raise LocalZipUnavailable(f"Could not read frame for {video_id}: {exc}") from exc
