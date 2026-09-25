"""
Build video_id -> zip byte-range manifest for the organizer's remote video
ZIPs, so app/services/remote_zip_proxy.py can stream playback straight from
https://aic-data.ledo.io.vn/... via HTTP Range requests, with no local
download and no re-parsing at request time.

Only reads ZIP central-directory + local-file-header bytes (a few requests
per archive, no video content) via a Range-backed file-like object handed to
stdlib zipfile.ZipFile, so ZIP64 handling comes for free instead of being
hand-rolled.

Usage:
    python -m scripts.build_zip_video_index \
        --urls-file ../../challenge_resources/data/zip_video_links.txt \
        --output ../../challenge_resources/data/zip_video_index.json
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import httpx

LOCAL_HEADER_SIZE = 30
LOCAL_HEADER_STRUCT = "<4s5H3L2H"  # sig, ver, flags, method, time, date, crc32, csize, usize, fname_len, extra_len
ZIP_STORED = 0


class RangeReadFile:
    """Seekable read-only file-like object backed by HTTP Range requests.

    Handed to zipfile.ZipFile so it can locate/parse the central directory
    (including ZIP64) without us reimplementing that binary format.
    """

    def __init__(self, client: httpx.Client, url: str):
        self._client = client
        self._url = url
        self._pos = 0
        head = client.head(url, follow_redirects=True)
        head.raise_for_status()
        self._size = int(head.headers["content-length"])

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            end = self._size - 1
        else:
            end = min(self._pos + size, self._size) - 1
        if end < self._pos:
            return b""
        resp = self._client.get(
            self._url,
            headers={"Range": f"bytes={self._pos}-{end}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data


def local_data_offset(client: httpx.Client, url: str, header_offset: int) -> int:
    """Byte offset where a ZIP_STORED entry's raw data starts (after its
    local file header, whose filename/extra lengths can differ from the
    central directory's)."""
    resp = client.get(
        url,
        headers={"Range": f"bytes={header_offset}-{header_offset + LOCAL_HEADER_SIZE - 1}"},
    )
    resp.raise_for_status()
    header = resp.content
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local file header at offset {header_offset} in {url}")
    fields = struct.unpack(LOCAL_HEADER_STRUCT, header)
    filename_len, extra_len = fields[9], fields[10]
    return header_offset + LOCAL_HEADER_SIZE + filename_len + extra_len


def index_zip(client: httpx.Client, url: str) -> dict[str, dict]:
    import zipfile

    entries: dict[str, dict] = {}
    zf = zipfile.ZipFile(RangeReadFile(client, url))
    for info in zf.infolist():
        if not info.filename.lower().endswith((".mp4", ".mov")):
            continue
        if info.compress_type != ZIP_STORED:
            print(f"  ! skip {info.filename}: not ZIP_STORED (compress_type={info.compress_type})")
            continue
        video_id = Path(info.filename).stem
        data_offset = local_data_offset(client, url, info.header_offset)
        entries[video_id] = {
            "zip_url": url,
            "entry_name": info.filename,
            "data_offset": data_offset,
            "size": info.file_size,
        }
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", type=Path, required=True, help="One zip URL per line")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    urls = [
        line.strip()
        for line in args.urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        print(f"No URLs found in {args.urls_file}", file=sys.stderr)
        sys.exit(1)

    index: dict[str, dict] = {}
    with httpx.Client(timeout=30.0) as client:
        for url in urls:
            print(f"[index] {url}")
            entries = index_zip(client, url)
            for video_id, entry in entries.items():
                if video_id in index:
                    print(f"  ! duplicate video_id {video_id} in {url}, keeping first ({index[video_id]['zip_url']})")
                    continue
                index[video_id] = entry
            print(f"  -> {len(entries)} videos")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=None, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(index)} videos total -> {args.output}")


if __name__ == "__main__":
    main()
