"""Phase 2 orchestration: GPU consumer loop over CPU-produced batches, and per-video finalize."""
from __future__ import annotations

import json
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import load_image_encoder
from .producer import ProducerFailure, StopSignal, TensorBatch, VideoEnd, embedding_producer
from ..io_utils import atomic_json, safe_video_id


def finalize_video_embeddings(
    args,
    entry_name: str,
    video_id: str,
    vectors: np.memmap,
    decode_preprocess_s: float,
    gpu_seconds: float,
    precision: str,
    embedding_dim: int,
) -> dict:
    video_out = args.out_dir / video_id
    keyframes_data = json.loads((video_out / "keyframes.json").read_text(encoding="utf-8"))
    selected = keyframes_data["keyframes"]
    vectors.flush()

    if not args.no_jsonl:
        with (video_out / "vectors.jsonl").open("w", encoding="utf-8") as f:
            for position, item in enumerate(selected):
                f.write(json.dumps({
                    "entry": entry_name,
                    "frame_number": item["frame_number"],
                    "scene_index": item["scene_index"],
                    "vector": np.asarray(vectors[position], dtype=np.float32).tolist(),
                }) + "\n")

    stats = {
        "entry": entry_name,
        "video_id": video_id,
        "status": "ok",
        "num_keyframes": len(selected),
        "embedding_dim": int(embedding_dim),
        "device": args.device,
        "precision": f"amp-{precision}" if precision != "off" else "fp32",
        "weights_dtype": "fp32",
        "normalize_dtype": "fp32",
        "tf32": bool(args.tf32),
        "compiled": bool(args.compile),
        "timing_seconds": {
            "cpu_decode_and_preprocess": round(decode_preprocess_s, 3),
            "gpu_embedding": round(gpu_seconds, 3),
        },
        "throughput": {
            "decode_preprocess_keyframes_per_s": round(len(selected) / decode_preprocess_s, 3) if decode_preprocess_s > 0 else None,
            "gpu_embedding_images_per_s": round(len(selected) / gpu_seconds, 3) if gpu_seconds > 0 else None,
        },
    }
    atomic_json(video_out / "embedding_stats.json", stats)
    return stats


def run_embedding_phase(
    args,
    entries: list,
    prior_failures: list[dict],
) -> tuple[list[dict], list[dict]]:
    pending = [
        entry for entry in entries
        if args.overwrite or not (args.out_dir / safe_video_id(entry.name) / "embeddings.npy").exists()
    ]
    if not pending:
        print("All embeddings already exist; PE-Core phase skipped.")
        return [], prior_failures

    model, preprocess, preprocess_plan, amp_mode = load_image_encoder(args)
    # `model.num_features` is timm's declared architecture width, which for attention-pooling
    # heads (e.g. PE-Core) is not necessarily the actual pooled output dimension -- and has been
    # observed to disagree with the real forward-pass output across timm versions. Probe it for
    # real instead of trusting a static attribute that can silently drift out from under us.
    probe_size = preprocess_plan.output_h if preprocess_plan is not None else 448
    with torch.inference_mode():
        probe_out = model(torch.zeros(1, 3, probe_size, probe_size, device=args.device, dtype=torch.float32))
    embedding_dim = int(probe_out.shape[-1])
    declared_dim = getattr(model, "num_features", None)
    if declared_dim is not None and int(declared_dim) != embedding_dim:
        print(
            f"[warning] model.num_features={declared_dim} disagrees with actual forward output "
            f"dim={embedding_dim}; using the probed value.", flush=True,
        )
    print(f"Encoder embedding_dim={embedding_dim}", flush=True)
    batch_queue: queue.Queue = queue.Queue(maxsize=args.prefetch_batches)
    producer = threading.Thread(
        target=embedding_producer,
        args=(args, pending, preprocess, preprocess_plan, batch_queue),
        name="keyframe-cpu-producer",
        daemon=True,
    )
    producer.start()

    states: dict[str, dict] = {}
    results: list[dict] = []
    failures = list(prior_failures)
    queue_wait_s = 0.0
    gpu_wall_s = 0.0
    batch_count = 0
    image_count = 0
    gpu_mean = gpu_std = None
    if preprocess_plan is not None:
        gpu_mean = torch.tensor(preprocess_plan.mean, device=args.device, dtype=torch.float32).view(1, 3, 1, 1)
        gpu_std = torch.tensor(preprocess_plan.std, device=args.device, dtype=torch.float32).view(1, 3, 1, 1)

    entry_by_video = {safe_video_id(e.name): e.name for e in pending}

    def get_state(video_id: str, entry_name: str) -> dict:
        state = states.get(video_id)
        if state is not None:
            return state
        keyframes_data = json.loads(
            (args.out_dir / video_id / "keyframes.json").read_text(encoding="utf-8")
        )
        vector_path = args.out_dir / video_id / "embeddings.partial.npy"
        vector_path.unlink(missing_ok=True)
        vectors = np.lib.format.open_memmap(
            vector_path, mode="w+", dtype=np.float32,
            shape=(int(keyframes_data["num_keyframes"]), embedding_dim),
        )
        state = {
            "vectors": vectors,
            "gpu_seconds": 0.0,
            "entry": entry_name,
            "expected": int(keyframes_data["num_keyframes"]),
            "processed": 0,
            "decode_preprocess_s": None,
            "end_received": False,
        }
        states[video_id] = state
        return state

    def maybe_finalize(video_id: str) -> None:
        state = states.get(video_id)
        if state is None or not state["end_received"] or state["processed"] != state["expected"]:
            return
        stats = finalize_video_embeddings(
            args, state["entry"], video_id, state["vectors"],
            float(state["decode_preprocess_s"] or 0.0), state["gpu_seconds"], amp_mode, embedding_dim,
        )
        results.append(stats)
        print(
            f"[embedded] {state['entry']}: {stats['num_keyframes']} keyframes "
            f"(CPU {float(state['decode_preprocess_s'] or 0.0):.1f}s, GPU {state['gpu_seconds']:.1f}s)",
            flush=True,
        )
        del state["vectors"]
        os.replace(
            args.out_dir / video_id / "embeddings.partial.npy",
            args.out_dir / video_id / "embeddings.npy",
        )
        del states[video_id]

    while True:
        wait_started = time.perf_counter()
        item = batch_queue.get()
        queue_wait_s += time.perf_counter() - wait_started
        try:
            if isinstance(item, StopSignal):
                break
            if isinstance(item, ProducerFailure):
                failure = {"entry": item.entry_name, "phase": "embedding-producer", "error": item.error}
                failures.append(failure)
                print(f"[producer error] {item.entry_name}: {item.error}", flush=True)
                continue

            if isinstance(item, TensorBatch):
                started = time.perf_counter()
                gpu_batch = item.tensors.to(
                    args.device, non_blocking=args.pin_memory and args.device.startswith("cuda"),
                )
                if gpu_batch.dtype == torch.uint8:
                    if gpu_mean is None or gpu_std is None:
                        raise RuntimeError("received uint8 encoder batch without preprocess normalization config")
                    gpu_batch = gpu_batch.float().div_(255.0).sub_(gpu_mean).div_(gpu_std)
                autocast_enabled = amp_mode != "off" and args.device.startswith("cuda")
                autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
                        features = model(gpu_batch)
                    # Normalize outside autocast in FP32 for stable persisted embeddings.
                    features = F.normalize(features.float(), dim=-1)
                vectors_np = features.detach().cpu().numpy()
                elapsed_gpu = time.perf_counter() - started
                gpu_wall_s += elapsed_gpu
                batch_count += 1
                image_count += len(item.video_ids)

                log_every = max(1, getattr(args, "log_every_batches", 5))
                if batch_count == 1 or batch_count % log_every == 0:
                    img_s = image_count / gpu_wall_s if gpu_wall_s > 0 else 0.0
                    print(
                        f"  [Embed GPU] batches={batch_count} images_embedded={image_count} "
                        f"gpu_img_s={img_s:.1f} queue={batch_queue.qsize()}",
                        flush=True,
                    )

                if vectors_np.shape[1] != embedding_dim:
                    raise ValueError(
                        f"Embedding dimension {vectors_np.shape[1]}, expected model.num_features={embedding_dim}"
                    )

                # Scatter a mixed global batch back into each video's row order.
                per_video: dict[str, list[int]] = {}
                for row, video_id in enumerate(item.video_ids):
                    per_video.setdefault(video_id, []).append(row)
                for video_id, rows in per_video.items():
                    entry_name = item.entry_names[rows[0]]
                    state = get_state(video_id, entry_name)
                    pos = item.positions[rows]
                    state["vectors"][pos] = vectors_np[rows]
                    state["processed"] += len(rows)
                    state["gpu_seconds"] += elapsed_gpu * (len(rows) / len(item.video_ids))
                    maybe_finalize(video_id)

                del gpu_batch, features, vectors_np, item.tensors

            elif isinstance(item, VideoEnd):
                state = get_state(item.video_id, item.entry_name)
                state["decode_preprocess_s"] = item.elapsed_decode_preprocess_s
                state["end_received"] = True
                maybe_finalize(item.video_id)
        except Exception as exc:
            entry_name = getattr(item, "entry_name", "<mixed-global-batch>")
            failure = {"entry": entry_name, "phase": "embedding-gpu", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[GPU error] {entry_name}: {failure['error']}", flush=True)
        finally:
            batch_queue.task_done()

    producer.join()
    # A clean producer/consumer run should have finalized every state.
    for video_id, state in list(states.items()):
        if state["end_received"] and state["processed"] == state["expected"]:
            maybe_finalize(video_id)
        else:
            failures.append({
                "entry": state["entry"],
                "phase": "embedding-finalize",
                "error": f"incomplete embeddings: {state['processed']}/{state['expected']}",
            })

    avg_batch = (image_count / batch_count) if batch_count else 0.0
    total_observed = queue_wait_s + gpu_wall_s
    wait_pct = (100.0 * queue_wait_s / total_observed) if total_observed else 0.0
    gpu_ips = (image_count / gpu_wall_s) if gpu_wall_s > 0 else 0.0
    print(
        f"[embed metrics] images={image_count} batches={batch_count} "
        f"avg_batch={avg_batch:.2f}/{args.batch_size} "
        f"gpu_wall={gpu_wall_s:.1f}s gpu_img_s={gpu_ips:.2f} "
        f"queue_wait={queue_wait_s:.1f}s ({wait_pct:.1f}%)",
        flush=True,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, failures
