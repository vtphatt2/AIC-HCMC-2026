"""
Resolve a video stored inside a zip archive to an ffmpeg `subfile` URL, so
ffmpeg can decode straight out of the zip -- no extraction step, no
intermediate file, no matter how large the archive or the video is.

Only works when the entry is ZIP_STORED (uncompressed). That's the normal
case for this video corpus: the videos are already h264/mp4-compressed, so
zipping them again buys nothing and archive tools store them raw. Verified
pixel-exact (max_abs_diff=0 over decoded frames) against extract-then-decode
for a real archive (Videos_L30_a.zip, 96 videos, 4.1GB, all ZIP_STORED).

If an archive instead used DEFLATE, this trick doesn't apply -- a compressed
entry has to be decompressed sequentially from the start, so there's no
byte-range to point ffmpeg at. Fall back to normal `zipfile.extract()` (one
video's worth of disk space, deleted after processing) for those.

Usage:
  # print the subfile URL for one entry:
  python zip_source.py --zip archive.zip --entry video/L30_V001.mp4

  # list video entries in an archive (name + whether the trick applies):
  python zip_source.py --zip archive.zip --list

As a library:
  from zip_source import subfile_url, iter_video_entries
  url = subfile_url(zip_path, "video/L30_V001.mp4")
  # pass `url` as --video to local_extract_frames.py / local_extract_keyframes.py
"""
from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path

LOCAL_HEADER_FIXED_SIZE = 30
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def subfile_url(zip_path: Path, entry_name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(entry_name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            f"{entry_name!r} is compressed (compress_type={info.compress_type}) inside {zip_path}; "
            "the zero-copy subfile trick only works for ZIP_STORED (uncompressed) entries -- "
            "fall back to zipfile.extract() for this one."
        )
    with open(zip_path, "rb") as f:
        f.seek(info.header_offset)
        header = f.read(LOCAL_HEADER_FIXED_SIZE)
    # Local file header (PK\x03\x04): fixed 30 bytes, then filename, then extra field.
    # Need filename_len/extra_len to find where the actual entry data starts --
    # info.header_offset only points at the start of this header, not the data.
    _, _, _, _, _, _, _, _, _, fname_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
    data_start = info.header_offset + LOCAL_HEADER_FIXED_SIZE + fname_len + extra_len
    data_end = data_start + info.file_size
    # Syntax confirmed against local ffmpeg 9.0: single comma between key/value
    # AND between pairs; double comma only around the whole option block and
    # before the final `:path`. (`subfile,,start,X,end,Y,,:path` -- other
    # comma arrangements either fail to parse or silently drop `start`.)
    return f"subfile,,start,{data_start},end,{data_end},,:{zip_path}"


def iter_video_entries(zip_path: Path) -> list[tuple[str, bool]]:
    """Return (entry_name, subfile_trick_applies) for every video-looking entry."""
    with zipfile.ZipFile(zip_path) as zf:
        return [
            (info.filename, info.compress_type == zipfile.ZIP_STORED)
            for info in zf.infolist()
            if info.filename.lower().endswith(VIDEO_EXTENSIONS)
        ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--entry", help="Entry name to resolve, e.g. video/L30_V001.mp4")
    p.add_argument("--list", action="store_true", help="List video entries instead of resolving one")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name, ok in iter_video_entries(args.zip):
            print(f"{'OK' if ok else 'COMPRESSED (no subfile trick)':<28} {name}")
        return
    if not args.entry:
        raise SystemExit("--entry is required unless --list is given")
    print(subfile_url(args.zip, args.entry))


if __name__ == "__main__":
    main()
