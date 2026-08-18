"""MP4 box / sample-table parsing, ported from the organizers'
remote_zip_video_toolkit (mp4_index.py) plus avcC (H.264 codec config)
parsing, which the toolkit didn't need for its own byte-range-only use case.

Operates on an already-fetched moov box (bytes) — no I/O here.
"""
from __future__ import annotations

import struct


def _u32(b: bytes, o: int) -> int:
    return struct.unpack_from(">I", b, o)[0]


def _u64(b: bytes, o: int) -> int:
    return struct.unpack_from(">Q", b, o)[0]


def iter_boxes(buf: bytes, start: int = 0, end: int | None = None):
    if end is None:
        end = len(buf)
    p = start
    while p + 8 <= end:
        size = _u32(buf, p)
        typ = buf[p + 4:p + 8]
        header = 8
        if size == 1:
            if p + 16 > end:
                break
            size = _u64(buf, p + 8)
            header = 16
        elif size == 0:
            size = end - p
        if size < header or p + size > end:
            break
        yield {
            "type": typ, "start": p, "size": size, "header": header,
            "payload_start": p + header, "end": p + size,
        }
        p += size


def find_child(buf: bytes, parent: dict, typ: bytes) -> dict | None:
    for box in iter_boxes(buf, parent["payload_start"], parent["end"]):
        if box["type"] == typ:
            return box
    return None


def find_children(buf: bytes, parent: dict, typ: bytes) -> list[dict]:
    return [b for b in iter_boxes(buf, parent["payload_start"], parent["end"]) if b["type"] == typ]


def parse_hdlr(buf: bytes, box: dict) -> bytes:
    p = box["payload_start"]
    return buf[p + 8:p + 12]


def parse_mdhd(buf: bytes, box: dict) -> tuple[int, int]:
    p = box["payload_start"]
    version = buf[p]
    if version == 1:
        return _u32(buf, p + 20), _u64(buf, p + 24)
    return _u32(buf, p + 12), _u32(buf, p + 16)


def parse_stts(buf: bytes, box: dict) -> list[tuple[int, int]]:
    p = box["payload_start"] + 4
    n = _u32(buf, p)
    p += 4
    out = []
    for _ in range(n):
        out.append((_u32(buf, p), _u32(buf, p + 4)))
        p += 8
    return out


def parse_ctts(buf: bytes, box: dict) -> list[tuple[int, int]]:
    p0 = box["payload_start"]
    version = buf[p0]
    p = p0 + 4
    n = _u32(buf, p)
    p += 4
    out = []
    for _ in range(n):
        count = _u32(buf, p)
        raw = _u32(buf, p + 4)
        offset = raw - (1 << 32) if (version == 1 and raw & 0x80000000) else raw
        out.append((count, offset))
        p += 8
    return out


def parse_stss(buf: bytes, box: dict) -> list[int]:
    p = box["payload_start"] + 4
    n = _u32(buf, p)
    p += 4
    return [_u32(buf, p + 4 * i) for i in range(n)]


def parse_stsz(buf: bytes, box: dict) -> list[int]:
    p = box["payload_start"] + 4
    sample_size = _u32(buf, p)
    p += 4
    sample_count = _u32(buf, p)
    p += 4
    if sample_size != 0:
        return [sample_size] * sample_count
    return [_u32(buf, p + 4 * i) for i in range(sample_count)]


def parse_stsc(buf: bytes, box: dict) -> list[tuple[int, int, int]]:
    p = box["payload_start"] + 4
    n = _u32(buf, p)
    p += 4
    out = []
    for _ in range(n):
        out.append((_u32(buf, p), _u32(buf, p + 4), _u32(buf, p + 8)))
        p += 12
    return out


def parse_chunk_offsets(buf: bytes, box: dict) -> list[int]:
    p = box["payload_start"] + 4
    n = _u32(buf, p)
    p += 4
    if box["type"] == b"stco":
        return [_u32(buf, p + 4 * i) for i in range(n)]
    return [_u64(buf, p + 8 * i) for i in range(n)]


def expand_stts(entries: list[tuple[int, int]], sample_count: int) -> list[int]:
    dts = [0] * sample_count
    cur = 0
    i = 0
    for count, delta in entries:
        for _ in range(count):
            if i >= sample_count:
                break
            dts[i] = cur
            cur += delta
            i += 1
    return dts


def expand_ctts(entries: list[tuple[int, int]], sample_count: int) -> list[int]:
    if not entries:
        return [0] * sample_count
    out = [0] * sample_count
    i = 0
    for count, off in entries:
        for _ in range(count):
            if i >= sample_count:
                break
            out[i] = off
            i += 1
    return out


def build_sample_offsets(
    sample_sizes: list[int],
    chunk_offsets: list[int],
    stsc_entries: list[tuple[int, int, int]],
) -> list[int]:
    sample_offsets = [0] * len(sample_sizes)
    sample_i = 0
    for j, (first_chunk, samples_per_chunk, _desc) in enumerate(stsc_entries):
        last_chunk = stsc_entries[j + 1][0] - 1 if j + 1 < len(stsc_entries) else len(chunk_offsets)
        for chunk_no in range(first_chunk, last_chunk + 1):
            chunk_off = chunk_offsets[chunk_no - 1]
            rel = 0
            for _ in range(samples_per_chunk):
                if sample_i >= len(sample_sizes):
                    return sample_offsets
                sample_offsets[sample_i] = chunk_off + rel
                rel += sample_sizes[sample_i]
                sample_i += 1
    return sample_offsets


def find_video_track(moov: bytes) -> tuple[dict, dict]:
    """Return (mdhd, stbl) for the first video track in moov."""
    root = {"payload_start": 8 if _u32(moov, 0) != 1 else 16, "end": len(moov)}
    for trak in find_children(moov, root, b"trak"):
        mdia = find_child(moov, trak, b"mdia")
        if not mdia:
            continue
        hdlr = find_child(moov, mdia, b"hdlr")
        if not hdlr or parse_hdlr(moov, hdlr) != b"vide":
            continue
        mdhd = find_child(moov, mdia, b"mdhd")
        minf = find_child(moov, mdia, b"minf")
        stbl = find_child(moov, minf, b"stbl") if minf else None
        if mdhd and stbl:
            return mdhd, stbl
    raise RuntimeError("No video track found in moov")


# VisualSampleEntry (avc1/hvc1) fixed fields after the 8-byte box header, before
# any child boxes (avcC/hvcC etc): SampleEntry.reserved[6]+data_reference_index[2]
# (8 bytes) + VisualSampleEntry's own fixed fields (70 bytes) = 78 bytes total.
_VISUAL_SAMPLE_ENTRY_FIXED_FIELDS = 78


def find_avc1_sample_entry(moov: bytes, stbl: dict) -> dict:
    stsd = find_child(moov, stbl, b"stsd")
    if not stsd:
        raise RuntimeError("stbl has no stsd box")
    p = stsd["payload_start"] + 8  # skip version/flags(4) + entry_count(4)
    size = _u32(moov, p)
    typ = moov[p + 4:p + 8]
    if typ not in (b"avc1", b"avc3"):
        raise RuntimeError(f"Unsupported video sample entry '{typ!r}' (only H.264 avc1/avc3 is supported)")
    return {
        "type": typ, "start": p, "size": size,
        "payload_start": p + 8 + _VISUAL_SAMPLE_ENTRY_FIXED_FIELDS,
        "end": p + size,
    }


def parse_avcc(moov: bytes, stbl: dict) -> tuple[list[bytes], list[bytes], int]:
    """Extract (sps_list, pps_list, nal_length_size) from the avcC box."""
    avc1 = find_avc1_sample_entry(moov, stbl)
    avcc = find_child(moov, avc1, b"avcC")
    if not avcc:
        raise RuntimeError("avc1 sample entry has no avcC box")
    data = moov[avcc["payload_start"]:avcc["end"]]
    nal_length_size = (data[4] & 0x03) + 1
    num_sps = data[5] & 0x1F
    p = 6
    sps_list = []
    for _ in range(num_sps):
        length = struct.unpack_from(">H", data, p)[0]
        p += 2
        sps_list.append(data[p:p + length])
        p += length
    num_pps = data[p]
    p += 1
    pps_list = []
    for _ in range(num_pps):
        length = struct.unpack_from(">H", data, p)[0]
        p += 2
        pps_list.append(data[p:p + length])
        p += length
    return sps_list, pps_list, nal_length_size
