"""List videos inside a ZIP and hand ffmpeg a source it can read directly.

ZIP_STORED (uncompressed) entries are handed to ffmpeg via its `subfile,,start,...,end,...,,:path`
protocol, so the video is decoded straight out of the ZIP with no extraction step. Compressed
entries are extracted to a temp file first (and removed afterwards) since ffmpeg cannot decode a
DEFLATEd entry through subfile.
"""
from __future__ import annotations

import contextlib
import re
import shutil
import struct
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
LOCAL_HEADER_FIXED_SIZE = 30


@dataclass(frozen=True)
class VideoEntry:
    name: str
    stored: bool
    size: int


def list_video_entries(zip_path: Path, pattern: str | None) -> list[VideoEntry]:
    regex = re.compile(pattern) if pattern else None
    with zipfile.ZipFile(zip_path) as zf:
        entries = [
            VideoEntry(info.filename, info.compress_type == zipfile.ZIP_STORED, info.file_size)
            for info in zf.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(VIDEO_EXTENSIONS)
            and (regex is None or regex.search(info.filename))
        ]
    return sorted(entries, key=lambda item: item.name)


def make_subfile_url(zip_path: Path, entry_name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(entry_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"Entry is compressed: {entry_name}")

    with zip_path.open("rb") as f:
        f.seek(info.header_offset)
        header = f.read(LOCAL_HEADER_FIXED_SIZE)
    if len(header) != LOCAL_HEADER_FIXED_SIZE:
        raise RuntimeError(f"Invalid ZIP local header: {entry_name}")

    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise RuntimeError(f"Bad ZIP local-header signature: {entry_name}")
    filename_len, extra_len = fields[-2], fields[-1]
    data_start = info.header_offset + LOCAL_HEADER_FIXED_SIZE + filename_len + extra_len
    data_end = data_start + info.file_size
    return f"subfile,,start,{data_start},end,{data_end},,:{zip_path.resolve()}"


@contextlib.contextmanager
def materialized_video(zip_path: Path, entry: VideoEntry, temp_root: Path) -> Iterator[str]:
    if entry.stored:
        yield make_subfile_url(zip_path, entry.name)
        return

    suffix = Path(entry.name).suffix or ".video"
    temp_path = temp_root / f"compressed_entry{suffix}"
    with zipfile.ZipFile(zip_path) as zf, zf.open(entry.name) as src, temp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
    try:
        yield str(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def run_checked(cmd: list[str], *, capture_stdout: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return result
