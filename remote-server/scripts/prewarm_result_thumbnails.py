"""Fill the existing JPEG disk cache with exact 640px search thumbnails.

Uses the already indexed frame numbers and the same decoder/resizer as the
live /api/zip-frame route. Safe to interrupt and rerun: cached frames are read
instead of decoded again. N frames that cannot be recovered by a short seek
are cached from one verified sequential pass per video; source videos and
search identities remain unchanged.

Run from remote-server:
    .venv/bin/python scripts/prewarm_result_thumbnails.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent
load_dotenv(SERVER_ROOT / ".env")
sys.path.insert(0, str(SERVER_ROOT))

from app.services import local_zip_media as media  # noqa: E402
from app.services.exact_frame_images import build_exact_images  # noqa: E402
from app.services.video_quarantine import excluded_video_ids  # noqa: E402


class SequentialFallback:
    """Coalesce failed N seeks into at most one offline pass per video/run."""

    def __init__(self, groups):
        self.groups = groups
        self.locks = defaultdict(asyncio.Lock)
        self.slots = asyncio.Semaphore(2)
        self.results = {}

    async def prepare(self, video, index):
        async with self.locks[video]:
            if video not in self.results:
                try:
                    async with self.slots:
                        await media.wait_for_live_card_idle()
                        print(f"Sequential thumbnail cache: {video}", flush=True)
                        await asyncio.to_thread(build_exact_images, video, index, self.groups[video])
                    self.results[video] = None
                except Exception as exc:
                    self.results[video] = exc
            if self.results[video] is not None:
                raise self.results[video]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path,
                        default=REPO_ROOT / "challenge_resources/data/vectors.meta.npz")
    parser.add_argument("--prefixes", default="NSML",
                        help="Archive prefix order; warm N, S, and M before legacy L cards.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only this many selected frames (pilot run).")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


async def warm_thumbnail(cache, key, produce, recover=None):
    """Yield to live traffic only before acquiring a cache decode lock.

    Recovery also waits for idle, so it must run after cache.get releases its
    lock. Otherwise a live request's heartbeat and that lock form a cycle.
    """
    await media.wait_for_live_card_idle()
    try:
        return await cache.get(key, produce)
    except media.LocalZipUnavailable:
        if recover is None:
            raise
    await recover()
    await media.wait_for_live_card_idle()
    return await cache.get(key, produce)


async def main():
    args = parse_args()
    if args.workers < 1 or args.progress_every < 1:
        raise ValueError("workers and progress-every must be positive")
    cache = media._card_disk_cache or media._disk_cache
    if cache is None:
        raise RuntimeError("ZIP_CARD_DISK_CACHE_DIR or ZIP_JPEG_DISK_CACHE_DIR must be configured")
    meta_stat = args.meta.stat()
    quarantined = excluded_video_ids()
    completion_marker = cache.directory / ".prewarm-result-thumbnails-complete"
    signature = json.dumps({
        "meta": str(args.meta.resolve()), "size": meta_stat.st_size,
        "mtime_ns": meta_stat.st_mtime_ns, "prefixes": args.prefixes,
        "cache_budget": cache.shard_budget * cache.SHARDS,
        "frame_cache_version": media.PRESENTATION_FRAME_CACHE_VERSION,
        "n_frame_cache_version": media.N_FRAME_CACHE_VERSION,
        "quarantined_videos": sorted(quarantined),
    }, sort_keys=True)
    if not args.limit and completion_marker.exists() and completion_marker.read_text() == signature:
        print("All selected thumbnails are already prewarmed.", flush=True)
        return
    with np.load(args.meta, allow_pickle=False) as meta:
        videos = meta["video_id"]
        frames = meta["frame_number"]
        timestamps = meta["timestamp_ms"]

    selected = [i for i, video in enumerate(videos)
                if str(video).startswith(tuple(args.prefixes)) and str(video) not in quarantined]
    priority = {prefix: rank for rank, prefix in enumerate(args.prefixes)}
    selected.sort(key=lambda row: priority.get(str(videos[row])[0], len(priority)))
    if args.limit:
        selected = selected[:args.limit]
    print(f"Prewarming {len(selected)} exact thumbnails with {args.workers} workers; "
          f"cache budget={cache.shard_budget * cache.SHARDS / 2**30:.1f} GiB", flush=True)

    queue: asyncio.Queue[int | None] = asyncio.Queue(maxsize=args.workers * 4)
    failures: list[tuple[str, int, str]] = []
    completed = 0
    reused = 0
    produced = 0
    start = time.monotonic()
    n_groups = defaultdict(list)
    for row in selected:
        if str(videos[row]).startswith("N"):
            n_groups[str(videos[row])].append(int(frames[row]))
    recovery = SequentialFallback(n_groups)

    async def worker():
        nonlocal completed, reused, produced
        while True:
            row = await queue.get()
            try:
                if row is None:
                    return
                video = str(videos[row])
                frame = int(frames[row])
                timestamp = int(timestamps[row])
                exact_frame = frame if video.startswith(("N", "S")) else None
                try:
                    index = await media._get_index(video)
                    key = media._cache_key(index, video, timestamp, 640, "jpeg", exact_frame)

                    async def produce():
                        original = await media._decode_frame_jpeg(
                            video, timestamp, 45.0, frame_number=exact_frame
                        )
                        return await asyncio.to_thread(media._resize_jpeg, original, 640)

                    if await cache.contains(key):
                        reused += 1
                    else:
                        await warm_thumbnail(cache, key, produce,
                                             (lambda: recovery.prepare(video, index))
                                             if video.startswith("N") else None)
                        produced += 1
                except Exception as exc:
                    failures.append((video, frame, str(exc)))
                completed += 1
                if completed % args.progress_every == 0 or completed == len(selected):
                    elapsed = time.monotonic() - start
                    print(f"{completed}/{len(selected)} frames; "
                          f"{completed / max(elapsed, .001):.1f} frames/s; "
                          f"reused={reused} new={produced} "
                          f"new_fps={produced / max(elapsed, .001):.1f}; "
                          f"failures={len(failures)}", flush=True)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(args.workers)]
    for row in selected:
        await queue.put(row)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers)
    if failures:
        failure_path = cache.directory / ".prewarm-result-thumbnails-failures.json"
        failure_path.write_text(json.dumps(failures, indent=2))
        for video, frame, error in failures[:30]:
            print(f"FAIL {video} frame={frame}: {error}", file=sys.stderr)
        print(f"{len(failures)} thumbnails could not be cached; details: {failure_path}",
              file=sys.stderr)
        # Preserve the report for review instead of retrying bad sources forever.
        raise SystemExit(2)
    if not args.limit:
        (cache.directory / ".prewarm-result-thumbnails-failures.json").unlink(missing_ok=True)
        completion_marker.write_text(signature)


if __name__ == "__main__":
    asyncio.run(main())
