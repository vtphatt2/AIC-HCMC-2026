"""Phase 1 orchestration: turn a list of VideoEntry into scenes.json/keyframes.json per video.

Two schedulers:
  - `_run_transnet_phase_sequential`: legacy v8 path, one video fully decoded then inferred at a
    time. Kept for correctness/performance A/B tests against the streaming-global path.
  - `_run_transnet_phase_global` (default): decode workers stream 100-frame windows from several
    videos concurrently into a bounded queue; a single GPU batcher mixes windows across videos so
    the GPU starts working after ~75 decoded frames instead of waiting for a whole video.
"""
from __future__ import annotations

import math
import queue
import shutil
import threading
import time

import numpy as np
import torch

from ..io_utils import atomic_json, safe_video_id
from .decode import (
    TransNetDecodedVideo,
    TransNetDecodeFailure,
    TransNetStreamDone,
    TransNetStreamStart,
    TransNetWindowItem,
    TRANSNET_H,
    TRANSNET_W,
    _decode_transnet_entry,
    _stream_transnet_entry,
)
from .model import _finalize_transnet_video, _to_numpy_prediction, _transnet_predict_raw_batch, load_transnet


def _run_transnet_phase_sequential(args, entries, model) -> tuple[list, list[dict]]:
    """Legacy v8 path, kept for correctness/performance A/B tests."""
    ready: list = []
    failures: list[dict] = []
    for index, entry in enumerate(entries, 1):
        video_id = safe_video_id(entry.name)
        video_out = args.out_dir / video_id
        keyframes_path = video_out / "keyframes.json"
        scenes_path = video_out / "scenes.json"
        if keyframes_path.exists() and scenes_path.exists() and not args.overwrite:
            print(f"[TransNet {index}/{len(entries)}] reuse {entry.name}", flush=True)
            ready.append(entry); continue
        print(f"[TransNet {index}/{len(entries)}] {entry.name}", flush=True)
        item = _decode_transnet_entry(args, entry)
        if isinstance(item, TransNetDecodeFailure):
            failure = {"entry": entry.name, "phase": "transnet", "error": item.error}
            failures.append(failure); print(f"  [error] {item.error}", flush=True); continue
        infer_started = time.perf_counter()
        try:
            # Keep the package helper for an exact legacy reference path.
            try:
                frames_arg = torch.from_numpy(item.frames).to(model.device) if hasattr(model, "device") else item.frames
                with torch.inference_mode():
                    predictions, _ = model.predict_frames(frames_arg, quiet=True)
            except TypeError:
                predictions, _ = model.predict_frames(item.frames)
            predictions_np = _to_numpy_prediction(predictions)[:len(item.frames)]
            infer_s = time.perf_counter() - infer_started
            _finalize_transnet_video(args, model, item, predictions_np, infer_s, math.ceil(len(item.frames)/50), math.ceil(len(item.frames)/50))
            ready.append(entry)
        except Exception as exc:
            failure = {"entry": entry.name, "phase": "transnet", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure); print(f"  [error] {failure['error']}", flush=True)
    return ready, failures


def _run_transnet_phase_global(args, entries, model) -> tuple[list, list[dict]]:
    """V9.2 true streaming global TransNet scheduler.

    ffmpeg workers emit temporal windows as soon as enough frames exist. The GPU
    batcher therefore starts after ~75 decoded frames instead of waiting for an
    entire 10k-20k-frame video (or a whole wave of videos) to finish decoding.
    """
    ready: list = []
    failures: list[dict] = []
    todo: list = []

    for index, entry in enumerate(entries, 1):
        video_id = safe_video_id(entry.name)
        video_out = args.out_dir / video_id
        if (video_out / "keyframes.json").exists() and (video_out / "scenes.json").exists() and not args.overwrite:
            print(f"[TransNet {index}/{len(entries)}] reuse {entry.name}", flush=True)
            ready.append(entry)
        else:
            if args.overwrite and video_out.exists():
                shutil.rmtree(video_out)
            todo.append(entry)

    if not todo:
        return ready, failures

    prefetch_windows = max(args.transnet_prefetch_windows, args.transnet_batch_size)
    print(
        f"[TransNet streaming-global] videos={len(todo)} batch={args.transnet_batch_size} "
        f"decode_workers={args.transnet_decode_workers} prefetch_windows={prefetch_windows} "
        f"batch_timeout_ms={args.transnet_batch_timeout_ms:g}", flush=True,
    )
    print("  GPU can start after the first ~75 decoded frames; full-video decode is NOT required.", flush=True)

    task_q: queue.Queue = queue.Queue()
    event_q: queue.Queue = queue.Queue(maxsize=prefetch_windows)
    for entry in todo:
        task_q.put(entry)
    for _ in range(args.transnet_decode_workers):
        task_q.put(None)

    def decode_worker() -> None:
        while True:
            entry = task_q.get()
            try:
                if entry is None:
                    return
                _stream_transnet_entry(args, entry, event_q)
            finally:
                task_q.task_done()

    workers = [
        threading.Thread(target=decode_worker, name=f"transnet-stream-decode-{i}", daemon=True)
        for i in range(args.transnet_decode_workers)
    ]
    for t in workers:
        t.start()

    states: dict[str, dict] = {}
    pending: list[TransNetWindowItem] = []
    terminal_videos = 0
    total_windows = 0
    total_gpu_s = 0.0
    total_useful_frames = 0
    total_batches = 0
    total_batch_items = 0
    total_queue_wait_s = 0.0
    gpu_idle_wait_s = 0.0
    max_queue_depth = 0
    phase_started = time.perf_counter()
    timeout_s = args.transnet_batch_timeout_ms / 1000.0

    def maybe_finalize(video_id: str) -> None:
        nonlocal terminal_videos
        state = states.get(video_id)
        if not state or state.get("finalized") or not state.get("decode_done"):
            return
        expected = math.ceil(state["num_frames"] / 50)
        if state["processed_windows"] != expected:
            return
        try:
            chunks = state["pred_chunks"]
            ordered = [chunks[start] for start in range(0, state["num_frames"], 50)]
            predictions_np = np.concatenate(ordered, axis=0)[:state["num_frames"]].astype(np.float32, copy=False)
            item = TransNetDecodedVideo(
                state["entry"], video_id, state["fps"], state["width"], state["height"],
                np.empty((0, TRANSNET_H, TRANSNET_W, 3), dtype=np.uint8), state["decode_seconds"],
            )
            _finalize_transnet_video(
                args, model, item, predictions_np,
                max(state["gpu_seconds"], 1e-9), state["batch_count"], state["batch_size_sum"],
                num_frames_override=state["num_frames"],
            )
            ready.append(state["entry"])
        except Exception as exc:
            failure = {"entry": state["entry"].name, "phase": "transnet_finalize", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"  [error] {failure['error']}", flush=True)
        state["finalized"] = True
        state["pred_chunks"].clear()
        terminal_videos += 1

    def fail_batch(batch_items: list[TransNetWindowItem], error: str) -> None:
        # A batch can mix windows from several videos; a GPU-side failure (e.g. CUDA OOM)
        # is not attributable to one of them, so mark every video in the batch as failed
        # and let the phase continue with the rest instead of crashing the whole process.
        nonlocal terminal_videos
        seen: set[str] = set()
        for item in batch_items:
            if item.video_id in seen:
                continue
            seen.add(item.video_id)
            state = states.get(item.video_id)
            if state is not None and not state.get("finalized"):
                state["finalized"] = True
                state["pred_chunks"].clear()
                terminal_videos += 1
            failures.append({"entry": item.entry.name, "phase": "transnet_gpu_batch", "error": error})
        print(f"  [GPU batch error] {len(seen)} video(s) affected: {error}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_pending(force: bool = False) -> None:
        nonlocal pending, total_windows, total_gpu_s, total_useful_frames, total_batches, total_batch_items
        while len(pending) >= args.transnet_batch_size or (force and pending):
            take = args.transnet_batch_size if len(pending) >= args.transnet_batch_size else len(pending)
            batch_items = pending[:take]
            del pending[:take]
            batch_np = np.stack([item.frames for item in batch_items], axis=0)
            try:
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_started = time.perf_counter()
                single, _many = _transnet_predict_raw_batch(model, batch_np, args.device)
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as exc:
                fail_batch(batch_items, f"{type(exc).__name__}: {exc}")
                del batch_np
                continue
            gpu_s = time.perf_counter() - gpu_started

            total_gpu_s += gpu_s
            total_batches += 1
            total_batch_items += take
            total_windows += take
            share = gpu_s / max(take, 1)
            for batch_i, item in enumerate(batch_items):
                state = states.get(item.video_id)
                if state is None:
                    raise RuntimeError(f"TransNet window arrived before stream start: {item.video_id}")
                state["pred_chunks"][item.output_start] = single[batch_i, 25:75, 0].astype(np.float32, copy=True)
                state["processed_windows"] += 1
                state["gpu_seconds"] += share
                state["batch_count"] += 1
                state["batch_size_sum"] += take
                # Every complete temporal window represents up to 50 useful output frames.
                if state.get("decode_done"):
                    useful = min(50, max(0, state["num_frames"] - item.output_start))
                else:
                    useful = 50
                total_useful_frames += useful
                maybe_finalize(item.video_id)

            log_every = max(1, getattr(args, "log_every_batches", 5))
            if total_batches == 1 or total_batches % log_every == 0:
                avg_b = total_batch_items / total_batches
                gpu_fps = total_useful_frames / max(total_gpu_s, 1e-9)
                print(
                    f"  [TransNet GPU] batches={total_batches} current_B={take} "
                    f"avg_B={avg_b:.1f}/{args.transnet_batch_size} useful_fps={gpu_fps:.1f} "
                    f"queue={event_q.qsize()} pending={len(pending)}",
                    flush=True,
                )
            del batch_np, single, _many

    while terminal_videos < len(todo):
        # If a full batch is ready, run it immediately without waiting for more events.
        if len(pending) >= args.transnet_batch_size:
            process_pending(force=False)
            continue

        wait_started = time.perf_counter()
        had_no_pending = not pending
        try:
            event = event_q.get(timeout=timeout_s if timeout_s > 0 else None)
            waited = time.perf_counter() - wait_started
            total_queue_wait_s += waited
            if had_no_pending:
                gpu_idle_wait_s += waited
            max_queue_depth = max(max_queue_depth, event_q.qsize())
        except queue.Empty:
            # Do not leave the GPU idle merely because a batch is partially filled.
            process_pending(force=True)
            continue

        if isinstance(event, TransNetStreamStart):
            states[event.video_id] = {
                "entry": event.entry, "fps": event.fps, "width": event.width, "height": event.height,
                "pred_chunks": {}, "processed_windows": 0, "gpu_seconds": 0.0,
                "batch_count": 0, "batch_size_sum": 0, "decode_done": False,
                "num_frames": None, "decode_seconds": None, "finalized": False,
            }
            print(f"  [stream start] {event.entry.name} ({event.width}x{event.height} @ {event.fps:.3f} fps)", flush=True)
        elif isinstance(event, TransNetWindowItem):
            pending.append(event)
            if len(pending) >= args.transnet_batch_size:
                process_pending(force=False)
        elif isinstance(event, TransNetStreamDone):
            state = states.get(event.video_id)
            if state is None:
                raise RuntimeError(f"TransNet done arrived before stream start: {event.video_id}")
            state["decode_done"] = True
            state["num_frames"] = event.num_frames
            state["decode_seconds"] = event.decode_seconds
            # Windows are provisionally counted as 50 useful frames before EOF is known.
            # Correct the last-window padding once the exact frame count arrives.
            total_useful_frames -= math.ceil(event.num_frames / 50) * 50 - event.num_frames
            print(
                f"  [decoded] {event.entry.name}: {event.num_frames} frames in {event.decode_seconds:.2f}s "
                f"({event.num_frames/max(event.decode_seconds,1e-9):.1f} fps)", flush=True,
            )
            maybe_finalize(event.video_id)
        elif isinstance(event, TransNetDecodeFailure):
            # If this stream had already started, mark it terminal and discard any partial state.
            state = states.get(event.video_id)
            if state is not None and not state.get("finalized"):
                state["finalized"] = True
                state["pred_chunks"].clear()
            failure = {"entry": event.entry.name, "phase": "transnet_decode", "error": event.error}
            failures.append(failure)
            terminal_videos += 1
            print(f"  [decode error] {event.entry.name}: {event.error}", flush=True)

    # Drain should normally be empty here; keep the invariant explicit.
    process_pending(force=True)
    for t in workers:
        t.join()

    phase_s = time.perf_counter() - phase_started
    avg_batch = total_batch_items / max(total_batches, 1)
    fill_ratio = avg_batch / args.transnet_batch_size if args.transnet_batch_size else 0.0
    print(
        f"[TransNet streaming summary] ready={len(ready)}/{len(entries)} batches={total_batches} "
        f"avg_batch={avg_batch:.2f}/{args.transnet_batch_size} fill={fill_ratio*100:.1f}% "
        f"GPU_useful_fps={total_useful_frames/max(total_gpu_s,1e-9):.1f} "
        f"wall_fps={total_useful_frames/max(phase_s,1e-9):.1f} "
        f"gpu_idle_wait={gpu_idle_wait_s:.1f}s wall={phase_s:.1f}s",
        flush=True,
    )
    atomic_json(args.out_dir / "transnet_performance_summary.json", {
        "mode": "streaming-global",
        "videos_total": len(entries),
        "videos_ready": len(ready),
        "videos_failed": len(failures),
        "configured_batch_size": args.transnet_batch_size,
        "decode_workers": args.transnet_decode_workers,
        "prefetch_windows": prefetch_windows,
        "batch_timeout_ms": args.transnet_batch_timeout_ms,
        "batches": total_batches,
        "average_batch_size": round(avg_batch, 3),
        "batch_fill_ratio": round(fill_ratio, 4),
        "windows": total_windows,
        "useful_output_frames": total_useful_frames,
        "gpu_seconds": round(total_gpu_s, 3),
        "gpu_useful_frames_per_second": round(total_useful_frames / max(total_gpu_s, 1e-9), 3),
        "queue_wait_seconds": round(total_queue_wait_s, 3),
        "gpu_idle_wait_seconds": round(gpu_idle_wait_s, 3),
        "max_event_queue_depth": max_queue_depth,
        "wall_seconds": round(phase_s, 3),
        "wall_useful_frames_per_second": round(total_useful_frames / max(phase_s, 1e-9), 3),
    })
    return ready, failures


def run_transnet_phase(args, entries) -> tuple[list, list[dict]]:
    model = load_transnet(args.device)
    try:
        if args.transnet_mode == "sequential":
            ready, failures = _run_transnet_phase_sequential(args, entries, model)
        else:
            ready, failures = _run_transnet_phase_global(args, entries, model)
        return ready, failures
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
