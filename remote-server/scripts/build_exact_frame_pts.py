"""Build exact decoded-frame timestamp maps for N/S search frames.

The compact maps let /api/zip-frame seek directly to the frame selected by the
preprocessing pipeline, even when MOV sample counts or timestamps differ from
FFmpeg's decoded output. Existing valid maps are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent
load_dotenv(SERVER_ROOT / ".env")
sys.path.insert(0, str(SERVER_ROOT))

from app.services import local_zip_media as media  # noqa: E402
from app.services.exact_frame_pts import build_exact_pts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path,
                        default=REPO_ROOT / "challenge_resources/data/vectors.meta.npz")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefixes", default="NS")
    parser.add_argument("--limit", type=int, default=0, help="Pilot: only this many videos")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    with np.load(args.meta, allow_pickle=False) as meta:
        groups: dict[str, list[int]] = defaultdict(list)
        for video, frame in zip(meta["video_id"], meta["frame_number"]):
            video_id = str(video)
            if video_id.startswith(tuple(args.prefixes)):
                groups[video_id].append(int(frame))
    ordered = sorted(groups, key=lambda video: (args.prefixes.find(video[0]), video))
    if args.limit:
        ordered = ordered[:args.limit]
    entries = media._scan_archives()
    unavailable = sorted(set(ordered) - entries.keys())
    if unavailable:
        raise RuntimeError(f"Missing source video for {len(unavailable)} IDs: {unavailable[:8]}")

    print(f"Building exact PTS maps for {len(ordered)} videos with {args.workers} workers", flush=True)
    started = time.monotonic()
    failures = []

    def build(video_id: str):
        index = media._build_index(video_id, entries[video_id])
        lot = ("S01" if video_id.startswith("S") else
               f"N{((int(video_id[1:4])-1)//10)*10+1:03d}-N{((int(video_id[1:4])-1)//10)*10+10:03d}")
        scenes = (REPO_ROOT / "challenge_resources/data/zip_embeddings" /
                  f"output_{lot}" / f"videos__{video_id}" / "scenes.json")
        frame_count = int(json.loads(scenes.read_text())["num_frames"])
        for attempt in range(3):
            try:
                return build_exact_pts(
                    video_id, index, groups[video_id], frame_count,
                    "/usr/bin/ffmpeg" if video_id.startswith("N") else media._ffmpeg_path(),
                    force_dense=attempt == 2,
                    store_full_timeline=video_id.startswith("N"),
                )
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(1)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(build, video_id): video_id for video_id in ordered}
        for number, future in enumerate(as_completed(pending), 1):
            video_id = pending[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((video_id, str(exc)))
                print(f"FAIL {video_id}: {exc}", file=sys.stderr, flush=True)
            if number % 10 == 0 or number == len(ordered):
                print(f"{number}/{len(ordered)} videos; failures={len(failures)}; "
                      f"elapsed={time.monotonic()-started:.0f}s", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} exact PTS maps failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
