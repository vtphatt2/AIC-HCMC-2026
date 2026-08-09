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
import subprocess
from bisect import bisect_right
from dataclasses import dataclass, field

import httpx

from app.services import mp4_box_parser as box
from app.services.remote_zip_proxy import RemoteZipVideoProxy, ZipVideoUnavailable


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


async def _fetch_range(client: httpx.AsyncClient, url: str, start: int, size: int) -> bytes:
    if size <= 0:
        return b""
    resp = await client.get(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    resp.raise_for_status()
    return resp.content


async def _find_moov(client: httpx.AsyncClient, url: str, mp4_offset: int, mp4_size: int) -> bytes:
    head_probe = min(1024 * 1024, mp4_size)
    head = await _fetch_range(client, url, mp4_offset, head_probe)
    for b in box.iter_boxes(head):
        if b["type"] == b"moov" and b["end"] <= len(head):
            return head[b["start"]:b["end"]]

    tail_size = min(4 * 1024 * 1024, mp4_size)
    tail_start_rel = mp4_size - tail_size
    tail = await _fetch_range(client, url, mp4_offset + tail_start_rel, tail_size)
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


async def _build_index(client: httpx.AsyncClient, video_id: str) -> VideoFrameIndex:
    entry = _proxy.lookup(video_id)
    if entry is None:
        raise ZipFrameUnavailable(f"No zip index entry for video_id={video_id}")

    moov = await _find_moov(client, entry["zip_url"], int(entry["data_offset"]), int(entry["size"]))

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


async def _get_index(client: httpx.AsyncClient, video_id: str) -> VideoFrameIndex:
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


async def get_frame_jpeg(
    client: httpx.AsyncClient,
    video_id: str,
    timestamp_ms: int,
    timeout_sec: float = 20.0,
) -> bytes:
    try:
        index = await _get_index(client, video_id)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise ZipFrameUnavailable(f"Could not fetch/parse moov for {video_id}: {exc}") from exc

    frame_id = round(max(0, int(timestamp_ms)) / 1000 * index.fps) if index.fps else 0
    start_sample, target_sample, end_sample = _range_plan(index, frame_id)
    region = index.samples[start_sample:end_sample + 1]
    region_start = min(s["o"] for s in region)
    region_end = max(s["o"] + s["s"] for s in region)

    fetched = await _fetch_range(
        client, index.zip_url, index.data_offset + region_start, region_end - region_start
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
    return await asyncio.to_thread(_decode_jpeg, bytes(stream), target_index, timeout_sec)
