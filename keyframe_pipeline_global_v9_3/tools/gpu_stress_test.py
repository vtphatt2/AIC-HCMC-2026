#!/usr/bin/env python3
"""Stress-test the TransNet or PE-Core GPU forward pass with synthetic input.

No video/ZIP/disk needed -- this loads the real model and feeds it random tensors
of the real input shape, so you can isolate GPU-side bugs (OOM, memory leak across
iterations) from decode/disk/data issues in minutes instead of waiting on a full
multi-hour run over real datasets.

Examples:
    # Does TransNet leak memory over many iterations at a fixed batch size?
    python tools/gpu_stress_test.py transnet --batch-size 16 --iters 200

    # Find the largest PE-Core batch size that doesn't OOM on this GPU.
    python tools/gpu_stress_test.py embed --ramp --iters 20

    # Same batch size run twice as separate processes -- confirms VRAM is fully
    # released between processes (the run_many.sh dataset-to-dataset scenario).
    python tools/gpu_stress_test.py transnet --batch-size 32 --iters 50
    python tools/gpu_stress_test.py transnet --batch-size 32 --iters 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch


def _mem_mb() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
    )


def run_batch_size(forward, make_batch, batch_size: int, iters: int, warmup: int, device: str) -> dict | None:
    """Run `iters` forward passes at this batch size. Returns stats, or None on OOM/error."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    try:
        for _ in range(warmup):
            forward(make_batch(batch_size))
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        alloc_by_iter = []
        t0 = time.perf_counter()
        for _ in range(iters):
            forward(make_batch(batch_size))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                alloc_by_iter.append(_mem_mb()[0])
        elapsed = time.perf_counter() - t0

        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else 0.0
        drift = (alloc_by_iter[-1] - alloc_by_iter[0]) if len(alloc_by_iter) > 1 else 0.0
        return {
            "batch_size": batch_size,
            "iters": iters,
            "elapsed_s": round(elapsed, 3),
            "items_per_s": round(batch_size * iters / elapsed, 2) if elapsed > 0 else None,
            "peak_alloc_mb": round(peak_alloc, 1),
            "alloc_drift_mb": round(drift, 1),
            "alloc_first_mb": round(alloc_by_iter[0], 1) if alloc_by_iter else None,
            "alloc_last_mb": round(alloc_by_iter[-1], 1) if alloc_by_iter else None,
        }
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  [OOM] batch_size={batch_size}: {exc}", flush=True)
        return None
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def stress(forward, make_batch, batch_sizes: list[int], iters: int, warmup: int, device: str) -> None:
    results = []
    for bs in batch_sizes:
        print(f"=== batch_size={bs} (warmup={warmup}, iters={iters}) ===", flush=True)
        stats = run_batch_size(forward, make_batch, bs, iters, warmup, device)
        if stats is None:
            print(f"batch_size={bs}: FAILED (OOM) -- largest working size so far: "
                  f"{results[-1]['batch_size'] if results else 'none'}")
            break
        results.append(stats)
        leak_flag = " <-- possible leak (VRAM grew across iterations)" if stats["alloc_drift_mb"] > stats["peak_alloc_mb"] * 0.05 else ""
        print(
            f"  items/s={stats['items_per_s']} peak_alloc={stats['peak_alloc_mb']}MB "
            f"alloc_drift={stats['alloc_drift_mb']}MB (first={stats['alloc_first_mb']}MB "
            f"last={stats['alloc_last_mb']}MB){leak_flag}",
            flush=True,
        )

    print("\n=== summary ===")
    print(f"{'batch':>6} {'items/s':>10} {'peak_MB':>10} {'drift_MB':>10}")
    for r in results:
        print(f"{r['batch_size']:>6} {r['items_per_s']:>10} {r['peak_alloc_mb']:>10} {r['alloc_drift_mb']:>10}")
    if not results:
        print("(no batch size succeeded)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("transnet", "embed"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=None, help="Single batch size to test")
    p.add_argument("--batch-sizes", default=None, help="Space-separated batch sizes to sweep, e.g. \"8 16 32 64\"")
    p.add_argument("--ramp", action="store_true", help="Auto-double batch size from 4 until OOM")
    p.add_argument("--iters", type=int, default=50, help="Forward passes per batch size (default 50)")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup passes before measuring (default 3)")
    p.add_argument("--embed-size", type=int, default=448, help="Synthetic PE-Core input H=W (default 448, ignored for transnet)")
    p.add_argument("--model-source", choices=("huggingface", "kaggle", "local"), default=None)
    p.add_argument("--kaggle-model", default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--amp", choices=("auto", "off", "bf16", "fp16"), default="auto")
    args = p.parse_args()

    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split()]
    elif args.batch_size:
        batch_sizes = [args.batch_size]
    elif args.ramp:
        batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    else:
        batch_sizes = [16]

    if args.phase == "transnet":
        from pipeline.phase1_transnet.decode import TRANSNET_H, TRANSNET_W
        from pipeline.phase1_transnet.model import _transnet_predict_raw_batch, load_transnet

        model = load_transnet(args.device)

        def make_batch(bs: int) -> np.ndarray:
            return np.random.randint(0, 256, size=(bs, 100, TRANSNET_H, TRANSNET_W, 3), dtype=np.uint8)

        def forward(batch_np: np.ndarray) -> None:
            _transnet_predict_raw_batch(model, batch_np, args.device)

        stress(forward, make_batch, batch_sizes, args.iters, args.warmup, args.device)

    else:
        from pipeline.phase2_embed.model import (
            DEFAULT_KAGGLE_MODEL, DEFAULT_MODEL_ARCH, DEFAULT_MODEL_ID, DEFAULT_MODEL_SOURCE,
            load_image_encoder,
        )
        ns = argparse.Namespace(
            local_files_only=False,
            model_source=args.model_source or DEFAULT_MODEL_SOURCE,
            model_id=DEFAULT_MODEL_ID,
            kaggle_model=args.kaggle_model or DEFAULT_KAGGLE_MODEL,
            model_arch=DEFAULT_MODEL_ARCH,
            model_dir=args.model_dir,
            kaggle_cache_dir=None,
            device=args.device,
            compile=False,
            precision=None,
            amp=args.amp,
            tf32=True,
            embed_ffmpeg_preprocess=False,  # skip building a real preprocess plan; synthetic input is already the target size
        )
        model, _preprocess, _plan, amp_mode = load_image_encoder(ns)
        autocast_enabled = amp_mode != "off" and args.device.startswith("cuda")
        autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
        size = args.embed_size

        def make_batch(bs: int) -> torch.Tensor:
            return torch.rand(bs, 3, size, size, device=args.device, dtype=torch.float32)

        def forward(batch: torch.Tensor) -> None:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                model(batch)

        stress(forward, make_batch, batch_sizes, args.iters, args.warmup, args.device)


if __name__ == "__main__":
    main()
