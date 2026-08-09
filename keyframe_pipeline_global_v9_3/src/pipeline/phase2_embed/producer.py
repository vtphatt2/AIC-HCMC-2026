"""CPU producer thread: decode requested keyframes, preprocess, and push GPU-sized batches.

Runs in its own thread so ffmpeg decode/preprocess for the next batch overlaps GPU inference of
the current one (see embed_phase.run_embedding_phase for the consumer side). The accumulator
intentionally survives video boundaries so the tail of video A can combine with the head of video
B into one full GPU batch, without decoding multiple videos concurrently.
"""
from __future__ import annotations

import json
import queue
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .decode import _decode_selected_frames, put_queue
from .preprocess import EncoderPreprocessPlan, normalize_uint8_cpu
from ..io_utils import safe_video_id
from ..video_probe import probe_video
from ..zip_source import VideoEntry, materialized_video


@dataclass
class TensorBatch:
    # A single GPU batch may contain keyframes from many different videos.
    tensors: torch.Tensor
    video_ids: list[str]
    entry_names: list[str]
    positions: np.ndarray
    frame_numbers: np.ndarray
    scene_indices: np.ndarray


@dataclass(frozen=True)
class VideoEnd:
    video_id: str
    entry_name: str
    elapsed_decode_preprocess_s: float


@dataclass(frozen=True)
class ProducerFailure:
    entry_name: str
    error: str
    detail: str


@dataclass(frozen=True)
class StopSignal:
    pass


STOP = StopSignal()


def embedding_producer(
    args,
    entries: list[VideoEntry],
    preprocess,
    preprocess_plan: EncoderPreprocessPlan | None,
    out_queue: queue.Queue,
) -> None:
    """Decode videos sequentially but form *global* PE-Core batches.

    The accumulator intentionally survives video boundaries, so the tail of video A
    can be combined with the beginning of video B.  This keeps GPU batches full
    without decoding multiple videos concurrently.
    """
    tensors: list[torch.Tensor] = []
    video_ids: list[str] = []
    entry_names: list[str] = []
    positions: list[int] = []
    frame_numbers: list[int] = []
    scene_indices: list[int] = []

    def flush_global() -> None:
        if not tensors:
            return
        batch = torch.stack(tensors)
        if args.pin_memory and args.device.startswith("cuda"):
            batch = batch.pin_memory()
        put_queue(out_queue, TensorBatch(
            tensors=batch,
            video_ids=list(video_ids),
            entry_names=list(entry_names),
            positions=np.asarray(positions, dtype=np.int64),
            frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
            scene_indices=np.asarray(scene_indices, dtype=np.int32),
        ))
        tensors.clear(); video_ids.clear(); entry_names.clear()
        positions.clear(); frame_numbers.clear(); scene_indices.clear()

    try:
        for index, entry in enumerate(entries, 1):
            video_id = safe_video_id(entry.name)
            done_path = args.out_dir / video_id / "embeddings.npy"
            if done_path.exists() and not args.overwrite:
                print(f"[Embed producer {index}/{len(entries)}] skip {entry.name}", flush=True)
                continue

            print(f"[Embed producer {index}/{len(entries)}] decode {entry.name}", flush=True)
            started = time.perf_counter()
            try:
                keyframes_data = json.loads(
                    (args.out_dir / video_id / "keyframes.json").read_text(encoding="utf-8")
                )
                selected = keyframes_data["keyframes"]
                if not selected:
                    put_queue(out_queue, VideoEnd(video_id, entry.name, time.perf_counter() - started))
                    continue

                by_frame = {
                    int(item["frame_number"]): (position, int(item["scene_index"]))
                    for position, item in enumerate(selected)
                }
                target_frames = sorted(by_frame)
                seen_frames: set[int] = set()

                with tempfile.TemporaryDirectory(prefix=f"embed-{video_id}-") as temp_name:
                    with materialized_video(args.zip_path, entry, Path(temp_name)) as source:
                        fps, _, _ = probe_video(args.ffprobe_bin, source)
                        for frame_index, tensor in _decode_selected_frames(
                            args, source, fps, target_frames, preprocess, preprocess_plan,
                        ):
                            if frame_index not in by_frame:
                                raise RuntimeError(f"unexpected decoded frame {frame_index}")
                            position, scene_index = by_frame[frame_index]
                            tensors.append(tensor)
                            video_ids.append(video_id)
                            entry_names.append(entry.name)
                            positions.append(position)
                            frame_numbers.append(frame_index)
                            scene_indices.append(scene_index)
                            seen_frames.add(frame_index)

                            if args.save_preprocessed:
                                save_dir = args.out_dir / video_id / "preprocessed"
                                save_dir.mkdir(parents=True, exist_ok=True)
                                saved_tensor = (
                                    normalize_uint8_cpu(tensor, preprocess_plan)
                                    if preprocess_plan is not None and tensor.dtype == torch.uint8
                                    else tensor
                                )
                                np.save(
                                    save_dir / f"frame_{frame_index:09d}.npy",
                                    saved_tensor.numpy().astype(np.float16),
                                )

                            if len(tensors) >= args.batch_size:
                                flush_global()

                missing = sorted(set(target_frames).difference(seen_frames))
                if missing:
                    raise RuntimeError(
                        f"ffmpeg ended before {len(missing)} requested keyframe(s); "
                        f"first missing frame={missing[0]}, last requested={target_frames[-1]}"
                    )

                # This marker may arrive before a not-yet-full global batch containing
                # this video's tail.  The consumer finalizes only after every expected
                # row has actually been embedded.
                put_queue(out_queue, VideoEnd(video_id, entry.name, time.perf_counter() - started))
            except Exception as exc:
                put_queue(out_queue, ProducerFailure(
                    entry_name=entry.name,
                    error=f"{type(exc).__name__}: {exc}",
                    detail=traceback.format_exc(),
                ))

        flush_global()
        put_queue(out_queue, STOP)
    except BaseException as exc:
        put_queue(out_queue, ProducerFailure(
            entry_name="<producer>",
            error=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        ))
        put_queue(out_queue, STOP)
