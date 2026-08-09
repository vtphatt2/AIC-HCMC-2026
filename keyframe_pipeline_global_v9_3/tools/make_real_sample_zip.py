#!/usr/bin/env python3
"""Build a small real-video sample ZIP from a remote dataset ZIP, without downloading the whole
thing -- stops as soon as N complete video entries have arrived, then repackages just those into
a fresh, standard ZIP (with a proper central directory) that bench_batch_size.sh/the pipeline can
open normally.

Reuses the same local-header byte-parsing approach as pipeline/download/stream_download.py
(ZIP_STORED entries only; the dataset ZIPs this repo targets are packed that way).

Example:
    python tools/make_real_sample_zip.py \
        --url https://aic-data.ledo.io.vn/Videos_L26_d.zip \
        --out sample.zip --videos 5
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.zip_source import LOCAL_HEADER_FIXED_SIZE, VIDEO_EXTENSIONS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--videos", type=int, default=5, help="Stop after this many complete video entries (default 5)")
    p.add_argument("--entry-regex", default=None)
    p.add_argument("--chunk-mb", type=int, default=8)
    return p.parse_args()


def scan_ready_entries(raw_path: Path, available: int, offset: int, regex, seen: set[str]) -> tuple[int, list[tuple[str, int, int]]]:
    """Mirror of stream_download.LocalHeaderParser.scan, returning (name, start, end) triples."""
    found: list[tuple[str, int, int]] = []
    with raw_path.open("rb") as f:
        while offset + 4 <= available:
            f.seek(offset)
            if f.read(4) != b"PK\x03\x04":
                break  # central directory, or not enough data yet
            if offset + LOCAL_HEADER_FIXED_SIZE > available:
                break
            f.seek(offset)
            hdr = f.read(LOCAL_HEADER_FIXED_SIZE)
            _, _, flags, compression, _, _, _, comp_size, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", hdr)
            header_end = offset + LOCAL_HEADER_FIXED_SIZE + name_len + extra_len
            if header_end > available:
                break
            f.seek(offset + LOCAL_HEADER_FIXED_SIZE)
            name = f.read(name_len).decode("utf-8", errors="replace")
            if flags & 0x08:
                raise RuntimeError(f"entry {name!r} uses a data descriptor; can't stream-parse boundaries")
            data_end = header_end + comp_size
            if data_end > available:
                break
            is_video = name.lower().endswith(VIDEO_EXTENSIONS)
            selected = regex is None or regex.search(name)
            if is_video and selected and compression == ZIP_STORED and name not in seen:
                seen.add(name)
                found.append((name, header_end, data_end))
            offset = data_end
    return offset, found


def main() -> None:
    args = parse_args()
    regex = re.compile(args.entry_regex) if args.entry_regex else None
    collected: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    offset = 0

    with tempfile.TemporaryDirectory(prefix="sample_zip_") as tmpdir:
        raw_path = Path(tmpdir) / "raw.partial"
        with requests.get(args.url, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            downloaded = 0
            with raw_path.open("wb") as f:
                for chunk in r.iter_content(args.chunk_mb << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    f.flush()
                    downloaded += len(chunk)
                    offset, found = scan_ready_entries(raw_path, downloaded, offset, regex, seen)
                    collected.extend(found)
                    print(f"\rdownloaded {downloaded/1024**2:.1f} MiB, videos found: {len(collected)}", end="", flush=True)
                    if len(collected) >= args.videos:
                        break
            print()
            r.close()

        if not collected:
            raise SystemExit("No video entries found before giving up -- ZIP may not be ZIP_STORED, or --videos is too high for what was downloaded.")

        collected = collected[:args.videos]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("rb") as src, ZipFile(args.out, "w") as zf:
            for name, start, end in collected:
                src.seek(start)
                zf.writestr(name, src.read(end - start), compress_type=ZIP_STORED)

    size_mb = args.out.stat().st_size / 1024**2
    print(f"Wrote {args.out}: {len(collected)} video(s), {size_mb:.1f} MiB total (only the needed bytes were downloaded)")


if __name__ == "__main__":
    main()
