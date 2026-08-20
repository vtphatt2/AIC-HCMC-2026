"""Serve frames and video playback from a **local** copy of the organizers'
`Videos_L*.zip` archives, without unpacking them.

Same sample-table design as `local-client/local-backend/app/services/zip_frame_source.py`,
with the fetch layer swapped: that one issues HTTP Range requests against the
organizers' host, this one seeks in a file on this machine. Everything that
module carries to survive the network — retry/backoff, the on-disk `moov`
cache, the GOP region cache, single-flight — is dropped here, because a seek
into a local file costs nothing to repeat. What stays is the parsed sample
index per video, which is CPU work, not I/O.

Entries inside the archives are stored uncompressed (`ZIP_STORED`), so byte N of
a video is byte `data_offset + N` of the archive and any range is one seek away.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import subprocess
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from app.services import mp4_box_parser as box

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_HEADER_SIZE = 30
LOCAL_HEADER_STRUCT = "<4s5H3L2H"
ZIP_STORED = 0
STREAM_CHUNK = 1024 * 1024
# How far past the target to look for frames that are presented before it.
# H.264 reordering depth is small; 32 is far past any real encoder setting.
REORDER_LOOKAHEAD = 32

# ffmpeg is the only real cost on this path; reading the bytes is a local seek.
_decode_semaphore = asyncio.Semaphore(int(os.getenv("ZIP_FRAME_DECODE_CONCURRENCY", "6")))


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
            if not info.filename.lower().endswith(".mp4"):
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
                    if not (name.endswith("/scenes.json") and "video__" in name):
                        continue
                    video_id = name.split("video__")[1].split("/")[0]
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
    timescale: int
    fps: float
    base_pts: int  # pts of the first *presented* frame — rarely 0 when B-frames reorder
    samples: list[dict]
    keyframe_samples: list[int]
    keyframe_frames: list[int] = field(default_factory=list)
    sps: list[bytes] = field(default_factory=list)
    pps: list[bytes] = field(default_factory=list)
    nal_length_size: int = 4

    def nearest_keyframe_sample(self, frame_id: int) -> int:
        i = bisect_right(self.keyframe_frames, frame_id) - 1
        return self.keyframe_samples[max(0, i)]


_index_cache: dict[str, VideoFrameIndex] = {}
_index_locks: dict[str, asyncio.Lock] = {}


def _find_moov(zip_path: Path, mp4_offset: int, mp4_size: int) -> bytes:
    head = _read(zip_path, mp4_offset, min(1024 * 1024, mp4_size))
    for b in box.iter_boxes(head):
        if b["type"] == b"moov" and b["end"] <= len(head):
            return head[b["start"]:b["end"]]

    tail_size = min(4 * 1024 * 1024, mp4_size)
    tail = _read(zip_path, mp4_offset + mp4_size - tail_size, tail_size)
    p = 4
    while True:
        i = tail.find(b"moov", p)
        if i < 0:
            break
        hdr = i - 4
        if hdr >= 0:
            size = int.from_bytes(tail[hdr:hdr + 4], "big")
            header = 8
            if size == 1 and hdr + 16 <= len(tail):
                size = int.from_bytes(tail[hdr + 8:hdr + 16], "big")
                header = 16
            if size >= header and hdr + size <= len(tail):
                return tail[hdr:hdr + size]
        p = i + 4
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
    samples = [{"o": offsets[i], "s": sizes[i], "p": pts[i]} for i in range(sample_count)]

    base_pts = min(pts) if pts else 0
    keyset = sorted(s - 1 for s in key_samples_1based)  # 0-based
    keyframe_frames = [
        int(round(((samples[s]["p"] - base_pts) / timescale) * fps)) if fps else s
        for s in keyset
    ]
    sps, pps, nal_length_size = box.parse_avcc(moov, stbl)

    return VideoFrameIndex(
        zip_path=zip_path,
        data_offset=data_offset,
        timescale=timescale,
        fps=fps,
        base_pts=base_pts,
        samples=samples,
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
    tail = range(start_sample, len(index.samples))
    target_sample = min(tail, key=lambda i: (abs(index.samples[i]["p"] - target_pts), i))
    target_p = index.samples[target_sample]["p"]

    # Every frame presented before the target must be decoded as well, or the
    # rank below counts a picture the decoder never emitted. Reordering depth is
    # small and bounded, so a fixed look-ahead covers it without scanning on.
    horizon = min(len(index.samples), target_sample + 1 + REORDER_LOOKAHEAD)
    last_needed = max(
        i for i in range(start_sample, horizon) if index.samples[i]["p"] <= target_p
    )
    end_sample = min(last_needed + reorder_margin, len(index.samples) - 1)
    target_index = sum(
        1 for i in range(start_sample, end_sample + 1) if index.samples[i]["p"] < target_p
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


def _decode_jpeg(annexb_stream: bytes, target_index: int, timeout_sec: float) -> bytes:
    command = [
        _ffmpeg_path(),
        "-loglevel", "error",
        "-f", "h264",
        "-i", "pipe:0",
        "-vf", f"select=eq(n\\,{target_index})",
        "-vsync", "0",
        "-frames:v", "1",
        "-q:v", "4",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, input=annexb_stream,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_sec, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalZipUnavailable("ffmpeg timed out decoding frame") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode(errors="replace")[:300] if exc.stderr else ""
        raise LocalZipUnavailable(f"ffmpeg failed to decode frame: {error}") from exc
    if not result.stdout:
        raise LocalZipUnavailable("ffmpeg produced no frame")
    return result.stdout


async def get_frame_jpeg(video_id: str, timestamp_ms: int, timeout_sec: float = 20.0) -> bytes:
    """JPEG bytes for the frame at `timestamp_ms`, or `LocalZipUnavailable` —
    every failure on this path funnels through that one type so the route never
    has to guess at a raw parse/subprocess error."""
    try:
        index = await _get_index(video_id)

        frame_id = round(max(0, int(timestamp_ms)) / 1000 * index.fps) if index.fps else 0
        start_sample, target_sample, end_sample, target_index = _range_plan(index, frame_id)
        region = index.samples[start_sample:end_sample + 1]
        region_start = min(s["o"] for s in region)
        region_end = max(s["o"] + s["s"] for s in region)

        # ponytail: no region/JPEG cache — a local seek is ~free where the
        # HTTP version needed one. Add an LRU here if repeated frames in one
        # grid measurably cost ffmpeg time.
        fetched = await asyncio.to_thread(
            _read, index.zip_path, index.data_offset + region_start, region_end - region_start,
        )

        stream = bytearray()
        for nal in index.sps:
            stream += b"\x00\x00\x00\x01" + nal
        for nal in index.pps:
            stream += b"\x00\x00\x00\x01" + nal
        for s in region:
            rel = s["o"] - region_start
            stream += _avcc_to_annexb(fetched[rel:rel + s["s"]], index.nal_length_size)

        async with _decode_semaphore:
            return await asyncio.to_thread(
                _decode_jpeg, bytes(stream), target_index, timeout_sec
            )
    except LocalZipUnavailable:
        raise
    except (OSError, RuntimeError, IndexError, ValueError, KeyError, struct.error) as exc:
        raise LocalZipUnavailable(f"Could not read frame for {video_id}: {exc}") from exc
