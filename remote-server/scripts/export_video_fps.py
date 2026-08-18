#!/usr/bin/env python3
"""Export `video_id -> fps` for every video in the ingest archives.

`timestamp_ms = frame_number / fps * 1000` is computed at ingest from each
video's `phase1_transnet/video__<id>/scenes.json`. The frame routes invert that,
so they have to invert with the *same* fps — a value derived from the MP4's own
moov can land on a neighbouring one (25.0 vs 29.97 vs a VFR average), and that
error grows with the timestamp instead of staying bounded.

Reading it per request costs a zip open each time and needs the archives to
still be there. This writes the answer once:

    python scripts/export_video_fps.py

    challenge_resources/data/video_fps.json   {"L21_V001": 30.0, ...}

Rerun after ingesting new lots. Serving falls back to scanning the archives,
then to the moov, when this file is absent — so it is an optimisation, not a
requirement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def collect(zip_dir: Path) -> dict[str, float]:
    archives = sorted(zip_dir.glob("*_results.zip"))
    if not archives:
        raise SystemExit(f"No *_results.zip found under {zip_dir}")

    fps_by_video: dict[str, float] = {}
    for archive_path in archives:
        found = 0
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not (name.endswith("/scenes.json") and "video__" in name):
                    continue
                video_id = name.split("video__")[1].split("/")[0]
                if video_id in fps_by_video:
                    continue
                try:
                    fps = float(json.loads(archive.read(name))["fps"])
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    print(f"  ! {video_id}: unreadable fps ({exc})", file=sys.stderr)
                    continue
                if fps <= 0:
                    print(f"  ! {video_id}: non-positive fps {fps}", file=sys.stderr)
                    continue
                fps_by_video[video_id] = fps
                found += 1
        print(f"  {archive_path.name}: {found} videos")
    return fps_by_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-dir", type=Path,
                        default=REPO_ROOT / "challenge_resources" / "data" / "zip_file")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "challenge_resources" / "data" / "video_fps.json")
    args = parser.parse_args()

    t0 = time.time()
    fps_by_video = collect(args.zip_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(sorted(fps_by_video.items())), indent=None, separators=(",", ":")),
        encoding="utf-8",
    )

    distinct: dict[float, int] = {}
    for fps in fps_by_video.values():
        distinct[fps] = distinct.get(fps, 0) + 1
    print(f"\n{len(fps_by_video)} videos -> {args.output}  ({time.time() - t0:.1f}s)")
    for fps, count in sorted(distinct.items()):
        print(f"  fps={fps:<20} {count} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
