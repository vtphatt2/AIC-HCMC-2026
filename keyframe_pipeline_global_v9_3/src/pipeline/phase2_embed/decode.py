"""Decode only the requested keyframes from a video for the embedding phase.

Two strategies, chosen per-video by `_choose_embed_decode_plan`:
  - `sequential`: one ffmpeg pass with a frame `select` filter up to the last requested keyframe;
    ffmpeg still decodes every frame internally but only emits the selected ones.
  - `seek`: nearby keyframes (within `--embed-seek-gap-seconds`) are grouped, and each group gets
    its own input-side `-ss` seek + short decode. Cheaper when keyframes are sparse.
`auto` (default) estimates decoded-frame cost for both and picks whichever is cheaper enough
(`--embed-seek-min-savings`), capped by `--embed-seek-max-groups` ffmpeg process startups.
"""
from __future__ import annotations

import queue
import subprocess
from typing import Iterator

import numpy as np
import torch
from PIL import Image

from .preprocess import EncoderPreprocessPlan, fast_preprocess_rgb
from ..video_probe import probe_video


def read_exact(pipe, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = pipe.read(size - len(data))
        if not chunk:
            return None if not data else bytes(data)
        data.extend(chunk)
    return bytes(data)


def put_queue(out_queue: queue.Queue, item) -> None:
    while True:
        try:
            out_queue.put(item, timeout=0.5)
            return
        except queue.Full:
            continue


def _select_expr(frame_numbers: list[int]) -> str:
    return "+".join(f"eq(n\\,{int(frame)})" for frame in frame_numbers)


def _group_keyframes(targets: list[int], max_gap_frames: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for frame in targets:
        if not groups or frame - groups[-1][-1] > max_gap_frames:
            groups.append([frame])
        else:
            groups[-1].append(frame)
    return groups


def _choose_embed_decode_plan(
    args, targets: list[int], fps: float,
) -> tuple[str, list[list[int]], int, int]:
    sequential_frames = targets[-1] + 1
    max_gap = max(0, int(round(args.embed_seek_gap_seconds * fps)))
    groups = _group_keyframes(targets, max_gap)
    grouped_frames = sum(group[-1] - group[0] + 1 for group in groups)

    if args.embed_decode_mode == "sequential":
        return "sequential", [targets], sequential_frames, grouped_frames
    if args.embed_decode_mode == "seek":
        return "seek", groups, sequential_frames, grouped_frames

    saving = 1.0 - (grouped_frames / max(1, sequential_frames))
    use_seek = saving >= args.embed_seek_min_savings and len(groups) <= args.embed_seek_max_groups
    return ("seek" if use_seek else "sequential"), groups, sequential_frames, grouped_frames


def _decode_selected_frames(
    args,
    source: str,
    fps: float,
    targets: list[int],
    preprocess,
    preprocess_plan: EncoderPreprocessPlan | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield requested frames only; non-keyframes never cross the ffmpeg stdout pipe."""
    mode, groups, seq_est, group_est = _choose_embed_decode_plan(args, targets, fps)
    print(
        f"  embed decode={mode} targets={len(targets)} groups={len(groups)} "
        f"est_frames={group_est if mode == 'seek' else seq_est}/{seq_est}",
        flush=True,
    )

    if mode == "sequential":
        decode_groups = [(0, targets)]
    else:
        decode_groups = [(group[0], [frame - group[0] for frame in group]) for group in groups]

    fallback_size: tuple[int, int] | None = None
    if preprocess_plan is None:
        _, fw, fh = probe_video(args.ffprobe_bin, source)
        fallback_size = (fw, fh)

    for group_index, (start_frame, relative_targets) in enumerate(decode_groups):
        vf_parts = [f"select='{_select_expr(relative_targets)}'"]
        if preprocess_plan is not None:
            vf_parts.append(preprocess_plan.ffmpeg_filter)
            output_w, output_h = preprocess_plan.output_w, preprocess_plan.output_h
        else:
            assert fallback_size is not None
            output_w, output_h = fallback_size
        vf = ",".join(vf_parts)

        cmd = [args.ffmpeg_bin, "-v", "error"]
        if mode == "seek":
            cmd += ["-ss", f"{start_frame / fps:.9f}"]
        cmd += [
            "-i", source, "-map", "0:v:0", "-an",
            "-vf", vf, "-vsync", "0", "-pix_fmt", "rgb24",
            "-frames:v", str(len(relative_targets)), "-f", "rawvideo", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = output_w * output_h * 3
        try:
            for local_index, relative_frame in enumerate(relative_targets):
                raw = read_exact(proc.stdout, frame_bytes)
                if raw is None or len(raw) != frame_bytes:
                    raise RuntimeError(
                        f"ffmpeg returned only {local_index}/{len(relative_targets)} selected frames "
                        f"for decode group {group_index}"
                    )
                absolute_frame = start_frame + relative_frame
                if preprocess_plan is not None:
                    tensor = fast_preprocess_rgb(raw, preprocess_plan)
                else:
                    array = np.frombuffer(raw, dtype=np.uint8).reshape(output_h, output_w, 3)
                    tensor = preprocess(Image.fromarray(array))
                yield absolute_frame, tensor
        finally:
            stderr = proc.stderr.read() if proc.stderr else b""
            returncode = proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if returncode != 0:
                raise RuntimeError(
                    f"ffmpeg keyframe decode failed ({returncode}):\n"
                    f"{stderr.decode(errors='replace')}"
                )
