"""
Stream video playback directly from the organizer's remote ZIP archives via
HTTP Range requests, translated against a precomputed zip_video_index.json
manifest (see scripts/build_zip_video_index.py).

No video content is ever downloaded or decoded server-side: the client's
Range request is translated into the matching byte range inside the upstream
ZIP and forwarded as-is, so the browser's own <video> element does the
seeking/buffering exactly like it would against a plain MP4 URL.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx


class ZipVideoUnavailable(RuntimeError):
    """No index entry for this video_id, or the upstream zip misbehaved."""


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "challenge_resources").is_dir():
            return parent
    raise ZipVideoUnavailable("Repository root could not be located")


def _default_index_path() -> Path:
    data_root = Path(os.getenv("AIC_SAMPLE_ROOT", _repo_root() / "challenge_resources" / "data"))
    return data_root / "zip_video_index.json"


def _parse_range(range_header: str | None, size: int) -> tuple[int, int]:
    """Parse a single-range 'bytes=start-end' Range header.

    Defaults to the whole file when absent (rare in practice — browsers
    issue an initial Range request for <video src=...>).
    """
    if not range_header:
        return 0, size - 1

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        raise ZipVideoUnavailable(f"Unsupported Range header: {range_header!r}")

    start_s, end_s = match.groups()
    if start_s == "" and end_s == "":
        raise ZipVideoUnavailable("Empty Range header")

    if start_s == "":
        suffix_len = int(end_s)
        start = max(0, size - suffix_len)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1

    end = min(end, size - 1)
    if start < 0 or start > end:
        raise ZipVideoUnavailable(f"Invalid range {start}-{end} for size {size}")
    return start, end


class RemoteZipVideoProxy:
    def __init__(self, index_path: Path | None = None):
        self._index_path = index_path or _default_index_path()
        self._index: dict[str, dict] = {}
        if self._index_path.is_file():
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._index)

    def lookup(self, video_id: str) -> dict | None:
        return self._index.get(video_id)

    async def open_range(
        self,
        client: httpx.AsyncClient,
        video_id: str,
        range_header: str | None,
    ) -> tuple[dict, httpx.Response]:
        """Resolve video_id + a client Range header to a streamed, still-open
        upstream response plus the response headers the caller should send.
        Caller must consume/close the returned httpx.Response."""
        entry = self.lookup(video_id)
        if entry is None:
            raise ZipVideoUnavailable(f"No zip index entry for video_id={video_id}")

        size = int(entry["size"])
        data_offset = int(entry["data_offset"])
        start, end = _parse_range(range_header, size)

        req = client.build_request(
            "GET",
            entry["zip_url"],
            headers={"Range": f"bytes={data_offset + start}-{data_offset + end}"},
        )
        resp = await client.send(req, stream=True)
        if resp.status_code != 206:
            await resp.aclose()
            raise ZipVideoUnavailable(
                f"Upstream zip did not honor Range request for {video_id} "
                f"(status={resp.status_code})"
            )

        headers = {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": "video/mp4",
        }
        return headers, resp
