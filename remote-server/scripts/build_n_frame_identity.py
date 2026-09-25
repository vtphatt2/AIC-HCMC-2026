"""Audit source-frame selection for the already indexed N vectors.

This replays the Phase-2 FFmpeg selection (without loading PE-Core), matches
each selected picture's pixel checksum to the verified full decoded timeline,
then writes a diagnostic candidate, never a serving map. Reproducing a seek
is not proof of the stored vector's identity; no indexed data is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent
load_dotenv(SERVER_ROOT / ".env")
sys.path.insert(0, str(SERVER_ROOT))
sys.path.insert(0, str(REPO_ROOT / "keyframe_pipeline_global_v9_3/src"))

from app.services import local_zip_media as media  # noqa: E402
from app.services.exact_frame_pts import index_path, showinfo_frames  # noqa: E402
from pipeline.phase2_embed.decode import _choose_embed_decode_plan, _select_expr  # noqa: E402


def reproduce_selected_frames(video_id: str, index, targets: list[int]) -> list[tuple[int, float]]:
    """Run the same auto/sequential/seek frame selection as Phase 2."""
    options = SimpleNamespace(
        embed_decode_mode="auto", embed_seek_gap_seconds=2.0,
        embed_seek_min_savings=0.30, embed_seek_max_groups=128,
    )
    mode, groups, _, _ = _choose_embed_decode_plan(options, targets, index.fps)
    source = (f"subfile,,start,{index.data_offset},end,"
              f"{index.data_offset + index.video_size},,:{index.zip_path}")
    selected_frames = []
    for group in groups:
        start = group[0] if mode == "seek" else 0
        relatives = [frame - start for frame in group]
        command = ["/usr/bin/ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info"]
        if mode == "seek":
            command += ["-ss", f"{start / index.fps:.9f}"]
        command += [
            "-i", source, "-map", "0:v:0", "-an",
            "-vf", f"select='{_select_expr(relatives)}',showinfo",
            "-vsync", "0", "-frames:v", str(len(group)), "-f", "null", "-",
        ]
        # FFmpeg can emit many warnings for malformed source MOVs, so drain
        # stderr while it runs; subprocess.run does that without a pipe stall.
        import subprocess
        result = subprocess.run(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=180)
        frames = showinfo_frames(result.stderr)
        bases = re.findall(rb"config in time_base:\s*(\d+)/(\d+)", result.stderr)
        if not bases or any((int(a), int(b)) != (1, index.timescale) for a, b in bases):
            raise RuntimeError(f"{video_id}: seek decode time base differs from 1/{index.timescale}")
        if result.returncode or len(frames) != len(group):
            raise RuntimeError(
                f"{video_id}: Phase-2 selection produced {len(frames)}/{len(group)} "
                f"frames (FFmpeg exit {result.returncode}) for {group[:3]}"
            )
        seek_seconds = start / index.fps if mode == "seek" else 0.0
        selected_frames.extend((checksum, seek_seconds + pts / index.timescale)
                               for _, pts, checksum in frames)
    if len(selected_frames) != len(targets):
        raise RuntimeError(f"{video_id}: selected frame count differs from keyframes")
    return selected_frames


def match_selected_frames(targets: list[int], selected: list[tuple[int, float]],
                          table: np.ndarray, fps: float,
                          timescale: int) -> tuple[dict[str, list[int]], list[int]]:
    """Match exact pixels, then exact PTS when a damaged seek changes pixels."""
    by_checksum: dict[int, list[int]] = defaultdict(list)
    for frame, checksum in enumerate(table[:, 2]):
        by_checksum[int(checksum)].append(frame)
    result = {}
    pixel_mismatches = []
    relative_pts = table[:, 1] - table[0, 1]
    if np.any(relative_pts[1:] < relative_pts[:-1]):
        raise ValueError("N decoded frame PTS is not monotonic")
    for target, (checksum, selected_seconds) in zip(targets, selected, strict=True):
        candidates = by_checksum.get(checksum)
        if not candidates:
            # A seek into a damaged GOP can decode its first picture with a
            # different checksum. FFmpeg still reports its presentation PTS;
            # require that to match the full-decode timeline within 2 ms.
            selected_pts = round(selected_seconds * timescale)
            position = int(np.searchsorted(relative_pts, selected_pts))
            nearby = range(max(0, position - 2), min(len(table), position + 3))
            actual = min(nearby, key=lambda frame: abs(int(relative_pts[frame]) - selected_pts))
            if abs(int(relative_pts[actual]) - selected_pts) > max(1, round(timescale * .002)):
                raise ValueError(f"No decoded source frame matches pixels or PTS for {target}")
            pixel_mismatches.append(target)
        else:
            nominal_pts = int(table[0, 1]) + round(target / fps * timescale)
            actual = min(candidates, key=lambda frame: (abs(int(table[frame, 1]) - nominal_pts), frame))
        relative_ms = round((int(table[actual, 1]) - int(table[0, 1])) * 1000 / timescale)
        result[str(target)] = [actual, relative_ms]
    actual_frames = [pair[0] for pair in result.values()]
    if any(next_frame < frame for frame, next_frame in zip(actual_frames, actual_frames[1:])):
        raise ValueError("Selected source frame order is not monotonic")
    return result, pixel_mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path,
                        default=REPO_ROOT / "challenge_resources/data/vectors.meta.npz")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--video-id", action="append", default=[],
                        help="Pilot: map only the named video; may be repeated.")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "challenge_resources/data/zip_embeddings/n_frame_identity.pending.json",
                        help="Write a diagnostic candidate; publication requires separate validation.")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    with np.load(args.meta, allow_pickle=False) as meta:
        groups: dict[str, list[int]] = defaultdict(list)
        for video, frame in zip(meta["video_id"], meta["frame_number"]):
            if str(video).startswith("N"):
                groups[str(video)].append(int(frame))
    ordered = sorted(groups)
    if args.video_id:
        unavailable = sorted(set(args.video_id) - groups.keys())
        if unavailable:
            raise ValueError(f"Video IDs absent from vector metadata: {unavailable}")
        ordered = sorted(set(args.video_id))
    if args.limit:
        ordered = ordered[:args.limit]
    entries = media._scan_archives()
    missing = sorted(set(ordered) - entries.keys())
    if missing:
        raise RuntimeError(f"Missing N video source: {missing[:5]}")
    started = time.monotonic()
    print(f"Mapping {len(ordered)} N videos, {sum(len(groups[v]) for v in ordered)} "
          f"search vectors, with {args.workers} workers", flush=True)

    def one_video(video_id: str):
        index = media._build_index(video_id, entries[video_id])
        lot_start = ((int(video_id[1:4]) - 1) // 10) * 10 + 1
        keyframes = (REPO_ROOT / "challenge_resources/data/zip_embeddings" /
                     f"output_N{lot_start:03d}-N{lot_start + 9:03d}" /
                     f"videos__{video_id}" / "keyframes.json")
        stored = [int(row["frame_number"]) for row in json.loads(keyframes.read_text())["keyframes"]]
        targets = groups[video_id]
        if stored != targets:
            raise RuntimeError(f"{video_id}: vector metadata/keyframe order differs")
        path = index_path(video_id, index)
        table = np.load(path, mmap_mode="r", allow_pickle=False)
        if table.ndim != 2 or table.shape[1] != 3 or not np.array_equal(table[:, 0], np.arange(len(table))):
            raise ValueError(f"{video_id}: incomplete exact decoded-frame map")
        selected = reproduce_selected_frames(video_id, index, targets)
        return match_selected_frames(targets, selected, table, index.fps, index.timescale)

    mapped = {}
    pixel_mismatches = {}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one_video, video_id): video_id for video_id in ordered}
        for number, future in enumerate(as_completed(jobs), 1):
            video_id = jobs[future]
            try:
                mapped[video_id], pixel_mismatches[video_id] = future.result()
            except Exception as exc:
                failures.append((video_id, str(exc)))
                print(f"FAIL {video_id}: {exc}", file=sys.stderr, flush=True)
            if number % 20 == 0 or number == len(ordered):
                print(f"{number}/{len(ordered)} videos; failures={len(failures)}; "
                      f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    payload = {"version": 1, "frames": dict(sorted(mapped.items())),
               "failures": dict(failures),
               "pixel_mismatches": {video: frames for video, frames in pixel_mismatches.items()
                                    if frames}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=args.output.parent, suffix=".json",
                                         delete=False) as file:
            temporary = file.name
            json.dump(payload, file, separators=(",", ":"))
        os.replace(temporary, args.output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    shifts = [actual - int(original) for video in mapped.values()
              for original, (actual, _) in video.items()]
    print(f"Diagnostic {args.output}: {len(shifts)} vectors, "
          f"shifted={sum(shift != 0 for shift in shifts)}, "
          f"seek_pixel_mismatches={sum(map(len, pixel_mismatches.values()))}, "
          f"maximum shift={max(map(abs, shifts), default=0)} frames", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
