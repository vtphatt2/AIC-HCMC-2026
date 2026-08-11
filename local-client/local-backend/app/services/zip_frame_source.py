"""Precise, single-Range-request frame extraction from the organizer's
remote ZIP — the real fix for app/services/zip_video_frame.py's original
approach (pointing ffmpeg at a URL and letting it probe the container,
which costs several round trips per frame and falls over under concurrent
load).

Ported from the organizers' remote_zip_video_toolkit's index-based design
(MP4IndexBuilder / IndexedRemoteVideo): build a per-video sample table once
(moov box only, cached in memory), then every frame request computes the
exact compressed byte range to fetch — one Range GET — and decodes it
directly. The toolkit explicitly leaves decoding as an integration detail
("Experimental"); that decode step (AVCC -> Annex-B, feeding raw H.264 to
ffmpeg) is implemented here.
"""
from __future__ import annotations

import asyncio
import base64
import os
import struct
import subprocess
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.services import mp4_box_parser as box
from app.services.range_http_client import RangeFetchError, RangeHTTPClient
from app.services.remote_zip_proxy import RemoteZipVideoProxy
# Same repo-root walk the storyboard cache uses — reused rather than copied a
# third time (text_encoder.py has its own).
from app.services.youtube_thumbnail import _find_repo_root


class ZipFrameUnavailable(RuntimeError):
    pass


@dataclass
class VideoFrameIndex:
    zip_url: str
    data_offset: int  # mp4 start offset within the zip
    timescale: int
    fps: float
    samples: list[dict]  # [{"o": offset, "s": size, "p": pts}, ...] within the mp4
    keyframe_samples: list[int]  # sorted sample indices that are keyframes
    keyframe_frames: list[int] = field(default_factory=list)  # matching frame_id per keyframe
    sps: list[bytes] = field(default_factory=list)
    pps: list[bytes] = field(default_factory=list)
    nal_length_size: int = 4

    def nearest_keyframe_sample(self, frame_id: int) -> int:
        i = bisect_right(self.keyframe_frames, frame_id) - 1
        return self.keyframe_samples[max(0, i)]


_index_cache: dict[str, VideoFrameIndex] = {}
_index_locks: dict[str, asyncio.Lock] = {}
_proxy = RemoteZipVideoProxy()

# ffmpeg is the only CPU-bound step here; everything else is waiting on Range
# requests. Capping the whole route at one number made a cold grid (many
# distinct videos, each needing a moov download first) queue those downloads
# behind ffmpeg processes. Cap the decodes here instead, and let the route
# allow far more requests in flight so the network work overlaps.
_decode_semaphore = asyncio.Semaphore(int(os.getenv("ZIP_FRAME_DECODE_CONCURRENCY", "6")))

# A grid asks for ~100 thumbnails at once, and they are not independent: the
# same frame often appears twice in one page of results, and frames that are
# seconds apart in the same video usually sit in the *same GOP*, so they need
# the identical byte range. Caching regions turns those into zero network work.
#
# Bounded by bytes rather than entries because regions vary a lot — a short GOP
# is ~100 KB, a long one ~800 KB.
_region_cache: "OrderedDict[tuple[str, int, int], bytes]" = OrderedDict()
_region_cache_bytes = 0
_region_cache_limit = int(os.getenv("ZIP_REGION_CACHE_MB", "192")) * 1024 * 1024
_region_locks: dict[tuple[str, int, int], asyncio.Lock] = {}


async def _fetch_region_cached(
    client: RangeHTTPClient, video_id: str, zip_url: str, absolute_start: int,
    region_start: int, length: int,
) -> bytes:
    """One Range request per region, however many frames want it.

    The lock matters as much as the cache: without it, a grid loading twelve
    thumbnails from one video fires twelve identical requests before any of
    them finishes and populates the cache."""
    global _region_cache_bytes
    key = (video_id, region_start, length)

    hit = _region_cache.get(key)
    if hit is not None:
        _region_cache.move_to_end(key)
        return hit

    lock = _region_locks.setdefault(key, asyncio.Lock())
    async with lock:
        hit = _region_cache.get(key)
        if hit is not None:
            _region_cache.move_to_end(key)
            return hit

        data = await client.fetch(zip_url, absolute_start, length)

        if len(data) <= _region_cache_limit:
            _region_cache[key] = data
            _region_cache_bytes += len(data)
            while _region_cache_bytes > _region_cache_limit and _region_cache:
                _, evicted = _region_cache.popitem(last=False)
                _region_cache_bytes -= len(evicted)
        _region_locks.pop(key, None)
        return data


async def _find_moov(client: RangeHTTPClient, url: str, mp4_offset: int, mp4_size: int) -> bytes:
    head_probe = min(1024 * 1024, mp4_size)
    head = await client.fetch(url, mp4_offset, head_probe)
    for b in box.iter_boxes(head):
        if b["type"] == b"moov" and b["end"] <= len(head):
            return head[b["start"]:b["end"]]

    tail_size = min(4 * 1024 * 1024, mp4_size)
    tail_start_rel = mp4_size - tail_size
    tail = await client.fetch(url, mp4_offset + tail_start_rel, tail_size)
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
    raise ZipFrameUnavailable("Could not locate a complete moov box (head/tail probe)")


def _moov_cache_path(video_id: str) -> Path:
    override = os.getenv("ZIP_MOOV_CACHE_DIR")
    path = Path(override) if override else _find_repo_root() / "cache" / "zip_moov"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{video_id}.moov"


async def _cached_moov(client: RangeHTTPClient, video_id: str, entry: dict) -> bytes:
    """The moov box is ~1 MB per video and is what makes a cold result grid
    slow: 30 distinct videos in one grid means ~30 MB pulled from the
    organizer's host before a single thumbnail can decode. It never changes,
    so pay for it once per video instead of once per backend restart —
    upstream bandwidth is the wall here, not concurrency."""
    cached = _moov_cache_path(video_id)
    try:
        if cached.is_file():
            return cached.read_bytes()
    except OSError:
        pass

    moov = await _find_moov(client, entry["zip_url"], int(entry["data_offset"]), int(entry["size"]))
    try:
        # Write-then-rename so a killed process can't leave a truncated moov
        # behind for the next run to parse.
        tmp = cached.with_suffix(".moov.part")
        tmp.write_bytes(moov)
        tmp.replace(cached)
    except OSError:
        pass
    return moov


async def _build_index(client: RangeHTTPClient, video_id: str) -> VideoFrameIndex:
    entry = _proxy.lookup(video_id)
    if entry is None:
        raise ZipFrameUnavailable(f"No zip index entry for video_id={video_id}")

    moov = await _cached_moov(client, video_id, entry)

    mdhd, stbl = box.find_video_track(moov)
    timescale, duration = box.parse_mdhd(moov, mdhd)

    stts_box = box.find_child(moov, stbl, b"stts")
    ctts_box = box.find_child(moov, stbl, b"ctts")
    stss_box = box.find_child(moov, stbl, b"stss")
    stsz_box = box.find_child(moov, stbl, b"stsz")
    stsc_box = box.find_child(moov, stbl, b"stsc")
    stco_box = box.find_child(moov, stbl, b"stco") or box.find_child(moov, stbl, b"co64")
    if not all([stts_box, stsz_box, stsc_box, stco_box]):
        raise ZipFrameUnavailable(f"{video_id}: required MP4 sample-table box missing")

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

    samples = [{"o": offsets[i], "s": sizes[i], "p": pts[i]} for i in range(sample_count)]

    keyset = sorted(s - 1 for s in key_samples_1based)  # 0-based
    keyframe_samples = keyset
    keyframe_frames = [
        int(round((samples[s]["p"] / timescale) * fps)) if fps else s
        for s in keyset
    ]

    sps, pps, nal_length_size = box.parse_avcc(moov, stbl)

    return VideoFrameIndex(
        zip_url=entry["zip_url"],
        data_offset=int(entry["data_offset"]),
        timescale=timescale,
        fps=fps,
        samples=samples,
        keyframe_samples=keyframe_samples,
        keyframe_frames=keyframe_frames,
        sps=sps,
        pps=pps,
        nal_length_size=nal_length_size,
    )


async def _get_index(client: RangeHTTPClient, video_id: str) -> VideoFrameIndex:
    cached = _index_cache.get(video_id)
    if cached is not None:
        return cached
    lock = _index_locks.setdefault(video_id, asyncio.Lock())
    async with lock:
        cached = _index_cache.get(video_id)
        if cached is not None:
            return cached
        index = await _build_index(client, video_id)
        _index_cache[video_id] = index
        return index


def _range_plan(index: VideoFrameIndex, frame_id: int, reorder_margin: int = 4) -> tuple[int, int, int]:
    """Returns (start_sample, target_sample, end_sample) — target_sample is
    the exact sample whose pts covers frame_id; end_sample adds a small
    margin so B-frame reordering has enough decoded lookahead."""
    target_pts = (frame_id / index.fps) * index.timescale if index.fps else 0
    start_sample = index.nearest_keyframe_sample(frame_id)
    target_sample = len(index.samples) - 1
    for i in range(start_sample, len(index.samples)):
        if index.samples[i]["p"] >= target_pts:
            target_sample = i
            break
    end_sample = min(target_sample + reorder_margin, len(index.samples) - 1)
    return start_sample, target_sample, end_sample


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
        raise ZipFrameUnavailable(
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
        raise ZipFrameUnavailable("ffmpeg timed out decoding frame") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode(errors="replace")[:300] if exc.stderr else ""
        raise ZipFrameUnavailable(f"ffmpeg failed to decode frame: {error}") from exc
    if not result.stdout:
        raise ZipFrameUnavailable("ffmpeg produced no frame")
    return result.stdout


async def get_frame_plan(
    http_client: httpx.AsyncClient,
    video_id: str,
    timestamp_ms: int,
) -> dict:
    """Everything a browser needs to decode this frame itself, and nothing else.

    Same work as get_frame_jpeg up to the point ffmpeg would start: locate the
    GOP, work out its byte range, and report where each sample sits inside it.
    The client then fetches that one range through /api/zip-bytes and runs
    WebCodecs on its own CPU.

    This exists because frame decoding is the one part of this backend that
    scales with *viewers* rather than with data — a shared result grid asks for
    up to 100 thumbnails per person, and every one of them was an ffmpeg
    subprocess on the host machine. Handing the decode to the client turns that
    into one Range fetch and a byte copy.

    Note what this does NOT do: point the browser at the organizer's host
    directly. That host serves no Access-Control-Allow-Origin, so a browser
    cannot read those bytes cross-origin however the request is framed. The
    bytes still travel through us; only the decoding leaves.
    """
    client = RangeHTTPClient(http_client)
    try:
        index = await _get_index(client, video_id)

        frame_id = round(max(0, int(timestamp_ms)) / 1000 * index.fps) if index.fps else 0
        start_sample, target_sample, end_sample = _range_plan(index, frame_id)
        region = index.samples[start_sample:end_sample + 1]
        region_start = min(s["o"] for s in region)
        region_end = max(s["o"] + s["s"] for s in region)

        sps = index.sps[0] if index.sps else b""
        if len(sps) < 4:
            raise ZipFrameUnavailable(f"No usable SPS for {video_id}")
        # avc1.PPCCLL — profile_idc, constraint flags, level_idc, straight out
        # of the SPS. VideoDecoder.configure() refuses to start without it.
        codec = "avc1." + "".join(f"{b:02x}" for b in sps[1:4])

        return {
            "video_id": video_id,
            "timestamp_ms": int(timestamp_ms),
            "codec": codec,
            "nal_length_size": index.nal_length_size,
            "timescale": index.timescale,
            "target_pts": region[target_sample - start_sample]["p"],
            "parameter_sets": [
                base64.b64encode(nal).decode() for nal in (*index.sps, *index.pps)
            ],
            "bytes_url": f"/api/zip-bytes/{video_id}/{region_start}/{region_end - region_start}",
            # Offsets relative to the fetched region, so the client can cut it
            # into samples without knowing anything about ZIP or MP4 layout.
            "samples": [
                {"o": s["o"] - region_start, "s": s["s"], "p": s["p"]} for s in region
            ],
        }
    except ZipFrameUnavailable:
        raise
    except RangeFetchError as exc:
        raise ZipFrameUnavailable(f"Could not build frame plan for {video_id}: {exc}") from exc
    except (RuntimeError, IndexError, ValueError, KeyError, struct.error) as exc:
        raise ZipFrameUnavailable(f"Could not parse MP4 structure for {video_id}: {exc}") from exc


async def fetch_region(
    http_client: httpx.AsyncClient,
    video_id: str,
    region_start: int,
    length: int,
) -> bytes:
    """Raw bytes for a region a plan already described. Deliberately not a
    general proxy: the range is validated against this video's own extent in
    the archive, so the endpoint can't be pointed at anything else."""
    client = RangeHTTPClient(http_client)
    entry = _proxy.lookup(video_id)
    if entry is None:
        raise ZipFrameUnavailable(f"No zip index entry for video_id={video_id}")
    size = int(entry["size"])
    if region_start < 0 or length <= 0 or region_start + length > size:
        raise ZipFrameUnavailable(
            f"Range {region_start}+{length} is outside {video_id} (size {size})"
        )
    try:
        return await _fetch_region_cached(
            client, video_id, entry["zip_url"],
            int(entry["data_offset"]) + region_start, region_start, length,
        )
    except RangeFetchError as exc:
        raise ZipFrameUnavailable(f"Could not fetch bytes for {video_id}: {exc}") from exc


async def get_frame_jpeg(
    http_client: httpx.AsyncClient,
    video_id: str,
    timestamp_ms: int,
    timeout_sec: float = 20.0,
) -> bytes:
    """Guaranteed to either return JPEG bytes or raise ZipFrameUnavailable —
    every failure in this pipeline (index build, range fetch, decode) is our
    own proxy's to own and funnels through one typed exception, never a raw
    network/subprocess error left for the caller to guess at."""
    client = RangeHTTPClient(http_client)
    try:
        index = await _get_index(client, video_id)

        frame_id = round(max(0, int(timestamp_ms)) / 1000 * index.fps) if index.fps else 0
        start_sample, target_sample, end_sample = _range_plan(index, frame_id)
        region = index.samples[start_sample:end_sample + 1]
        region_start = min(s["o"] for s in region)
        region_end = max(s["o"] + s["s"] for s in region)

        fetched = await _fetch_region_cached(
            client, video_id, index.zip_url,
            index.data_offset + region_start, region_start, region_end - region_start,
        )

        stream = bytearray()
        for nal in index.sps:
            stream += b"\x00\x00\x00\x01" + nal
        for nal in index.pps:
            stream += b"\x00\x00\x00\x01" + nal
        for s in region:
            rel = s["o"] - region_start
            stream += _avcc_to_annexb(fetched[rel:rel + s["s"]], index.nal_length_size)

        target_index = target_sample - start_sample
        async with _decode_semaphore:
            return await asyncio.to_thread(_decode_jpeg, bytes(stream), target_index, timeout_sec)
    except ZipFrameUnavailable:
        raise
    except RangeFetchError as exc:
        raise ZipFrameUnavailable(f"Could not fetch frame bytes for {video_id}: {exc}") from exc
    except (RuntimeError, IndexError, ValueError, KeyError, struct.error) as exc:
        # Malformed/unexpected MP4 structure for this particular video (e.g.
        # mp4_box_parser raising a plain RuntimeError, or a bad byte offset)
        # — treat as unavailable rather than a 500, same as any other
        # "this source doesn't work" case the caller already handles.
        raise ZipFrameUnavailable(f"Could not parse MP4 structure for {video_id}: {exc}") from exc
