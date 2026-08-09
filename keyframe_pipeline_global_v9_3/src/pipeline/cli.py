"""Argument parsing and the end-to-end `process_video_zip_gpu.py` entrypoint.

Phase 1 (CPU streaming decode + GPU, TransNetV2):
    Decode 48x27 RGB frames from multiple videos concurrently, emit 100-frame
    temporal windows immediately, globally batch them on the GPU, and save:
      <out>/<video_id>/scenes.json
      <out>/<video_id>/keyframes.json

Phase 2 (CPU producer + GPU consumer):
    Load PE-Core only after all TransNet work is complete. A CPU producer streams
    each video once through ffmpeg, discards non-keyframes, preprocesses selected
    frames, forms batches, and puts them into a bounded queue. The main thread
    consumes batches on the GPU, writes embeddings, and immediately releases the
    image tensors. CPU decoding/preprocessing of the next batch overlaps GPU
    inference of the current batch.

No JPEG keyframe files are required. ZIP_STORED entries are decoded directly from
inside the ZIP using ffmpeg's subfile protocol. Compressed entries are extracted
one video at a time into a temporary directory and removed after use.

Example:
    python src/process_video_zip_gpu.py \
        --zip Videos_L30_a.zip \
        --out-dir output \
        --batch-size 32 \
        --prefetch-batches 3 \
        --amp auto \
        --tf32

Dependencies:
    ffmpeg and ffprobe on PATH
    pip install numpy pillow torch timm huggingface_hub transnetv2-pytorch
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from .phase2_embed.model import DEFAULT_KAGGLE_MODEL, DEFAULT_MODEL_ARCH, DEFAULT_MODEL_ID, DEFAULT_MODEL_SOURCE
from .phase2_embed.phase import run_embedding_phase
from .io_utils import atomic_json, safe_video_id
from .phase1_transnet.phase import run_transnet_phase
from .zip_source import list_video_entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--entry-regex", default=None)
    p.add_argument("--limit", type=int, default=None, help="Limit the global sorted video list before sharding")
    p.add_argument("--num-shards", type=int, default=1, help="Split the sorted video list into N deterministic shards (default: 1)")
    p.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to process (default: 0)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--transnet-mode", choices=("global", "sequential"), default="global",
        help="TransNet scheduling mode. global batches 100-frame windows across videos; sequential keeps the legacy one-video-at-a-time path.",
    )
    p.add_argument(
        "--transnet-batch-size", type=int, default=16,
        help="Number of independent 100-frame TransNet windows per GPU forward in global mode (default: 16).",
    )
    p.add_argument(
        "--transnet-decode-workers", type=int, default=4,
        help="Parallel ffmpeg workers that decode 48x27 TransNet frames (default: 4).",
    )
    p.add_argument(
        "--transnet-active-videos", type=int, default=8,
        help="Decoded videos mixed into one global TransNet scheduling wave (default: 8).",
    )
    p.add_argument(
        "--transnet-prefetch-videos", type=int, default=8,
        help="Maximum decoded videos waiting while GPU inference runs (default: 8).",
    )
    p.add_argument(
        "--transnet-prefetch-windows", type=int, default=256,
        help="Maximum 100-frame TransNet windows queued ahead of the GPU in streaming global mode (default: 256).",
    )
    p.add_argument(
        "--transnet-batch-timeout-ms", type=float, default=20.0,
        help="Flush a partial TransNet batch after this many milliseconds without enough queued windows (default: 20 ms).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--prefetch-batches",
        type=int,
        default=3,
        help="Maximum preprocessed CPU batches waiting for the GPU",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--amp", choices=("auto", "off", "bf16", "fp16"), default="auto",
        help="Mixed-precision autocast mode. auto prefers BF16 when supported, otherwise FP16 on CUDA; off keeps FP32.",
    )
    p.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True,
        help="Allow TF32 for remaining FP32 CUDA matmul/cuDNN ops when the GPU supports it (default: enabled).",
    )
    p.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False,
        help="Compile the PE-Core vision encoder with torch.compile(mode='reduce-overhead').",
    )
    p.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default=None,
        help=argparse.SUPPRESS,  # v4 compatibility: fp32->amp off, fp16/bf16->matching autocast
    )
    p.add_argument("--model-source", choices=("huggingface", "kaggle", "local"), default=DEFAULT_MODEL_SOURCE, help="Where to load the PE-Core vision tower from")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face timm model id for --model-source huggingface")
    p.add_argument("--kaggle-model", default=DEFAULT_KAGGLE_MODEL, help="Kaggle model handle for --model-source kaggle")
    p.add_argument("--model-arch", default=DEFAULT_MODEL_ARCH, help="timm architecture name used for local/Kaggle checkpoint loading")
    p.add_argument("--model-dir", type=Path, default=None, help="Local directory containing model.safetensors or pytorch_model.bin for --model-source local")
    p.add_argument("--kaggle-cache-dir", type=Path, default=None, help="Optional kagglehub cache directory")
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download PE-Core from Hugging Face; require it to exist in the local HF cache",
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin prepared CPU batches before asynchronous GPU transfer",
    )
    p.add_argument(
        "--embed-ffmpeg-preprocess", action=argparse.BooleanOptionalAction, default=True,
        help="Let ffmpeg resize/center-crop selected keyframes from the encoder data config (default: enabled).",
    )
    p.add_argument(
        "--embed-decode-mode", choices=("auto", "sequential", "seek"), default="auto",
        help=("PE-Core CPU decode strategy. sequential decodes once up to the last keyframe; "
              "seek decodes sparse keyframe groups independently; auto chooses the cheaper estimate (default: auto)."),
    )
    p.add_argument(
        "--embed-seek-gap-seconds", type=float, default=2.0,
        help="In seek mode, keyframes separated by at most this many seconds share one ffmpeg decode group (default: 2.0).",
    )
    p.add_argument(
        "--embed-seek-min-savings", type=float, default=0.30,
        help="Auto mode uses grouped seeking only when estimated decoded frames are at least this fraction cheaper than sequential (default: 0.30).",
    )
    p.add_argument(
        "--embed-seek-max-groups", type=int, default=128,
        help="Auto mode refuses grouped seeking above this many ffmpeg groups to avoid process startup overhead (default: 128).",
    )
    p.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save model-input tensors as float16 .npy files (large; disabled by default)",
    )
    p.add_argument("--log-every-batches", type=int, default=5, help="Print GPU progress every N batches, for both phases (default 5; lower = more frequent updates)")
    p.add_argument("--no-jsonl", action="store_true")
    p.add_argument("--save-transnet-predictions", action="store_true", help="Save per-frame TransNet predictions.npy (disabled by default to save disk)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--phase", choices=("all", "transnet", "embed"), default="all",
        help="Run the full pipeline, only TransNet metadata, or only PE-Core embedding",
    )
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    return p.parse_args()


def check_environment(args: argparse.Namespace) -> None:
    if not args.zip_path.is_file():
        raise FileNotFoundError(args.zip_path)
    for executable in (args.ffmpeg_bin, args.ffprobe_bin):
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise FileNotFoundError(f"Executable not found: {executable}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.prefetch_batches <= 0:
        raise ValueError("--prefetch-batches must be positive")
    if args.embed_seek_gap_seconds < 0:
        raise ValueError("--embed-seek-gap-seconds must be >= 0")
    if not 0.0 <= args.embed_seek_min_savings < 1.0:
        raise ValueError("--embed-seek-min-savings must be in [0, 1)")
    if args.embed_seek_max_groups <= 0:
        raise ValueError("--embed-seek-max-groups must be positive")
    if args.transnet_batch_size <= 0:
        raise ValueError("--transnet-batch-size must be positive")
    if args.transnet_decode_workers <= 0:
        raise ValueError("--transnet-decode-workers must be positive")
    if args.transnet_active_videos <= 0:
        raise ValueError("--transnet-active-videos must be positive")
    if args.transnet_prefetch_videos <= 0:
        raise ValueError("--transnet-prefetch-videos must be positive")
    if args.transnet_prefetch_windows <= 0:
        raise ValueError("--transnet-prefetch-windows must be positive")
    if args.transnet_batch_timeout_ms < 0:
        raise ValueError("--transnet-batch-timeout-ms must be non-negative")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards - 1}]")


def run_summary_path(args: argparse.Namespace) -> Path:
    if args.num_shards == 1:
        return args.out_dir / "run_summary.json"
    return args.out_dir / f"run_summary.shard_{args.shard_index}_of_{args.num_shards}.json"


def summary_payload(args: argparse.Namespace, ready, embedding_results, failures) -> dict:
    return {
        "zip": str(args.zip_path),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "transnet_ready": [entry.name for entry in ready],
        "embedding_results": embedding_results,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    check_environment(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_entries = list_video_entries(args.zip_path, args.entry_regex)
    if args.limit is not None:
        all_entries = all_entries[:args.limit]
    if not all_entries:
        raise SystemExit("No matching videos found in ZIP")

    entries = all_entries[args.shard_index::args.num_shards]
    if args.num_shards > 1:
        print(
            f"Shard {args.shard_index}/{args.num_shards}: assigned {len(entries)} of "
            f"{len(all_entries)} selected video(s)", flush=True,
        )
    if not entries:
        summary = run_summary_path(args)
        atomic_json(summary, summary_payload(args, [], [], []))
        print(f"No videos assigned to this shard. Summary: {summary}", flush=True)
        return

    stored_count = sum(entry.stored for entry in entries)
    print(
        f"Found {len(entries)} video(s): {stored_count} ZIP_STORED/direct, "
        f"{len(entries) - stored_count} compressed/temp-extracted"
    )

    failures: list[dict] = []
    if args.phase in {"all", "transnet"}:
        print("\n=== PHASE 1/2: TransNet metadata for every video ===")
        ready, failures = run_transnet_phase(args, entries)
        summary = run_summary_path(args)
        atomic_json(summary, summary_payload(args, ready, [], failures))
        if args.phase == "transnet":
            print(f"\nDone. Summary: {summary}")
            if failures:
                raise SystemExit(1)
            return
    else:
        # Embed-only mode deliberately avoids loading TransNetV2. Only entries
        # with existing keyframes/scenes metadata are eligible.
        ready = []
        for entry in entries:
            video_out = args.out_dir / safe_video_id(entry.name)
            if (video_out / "keyframes.json").exists() and (video_out / "scenes.json").exists():
                ready.append(entry)
            else:
                failures.append({
                    "entry": entry.name,
                    "phase": "embed-prerequisite",
                    "error": "missing scenes.json/keyframes.json; run TransNet phase first",
                })
        print(f"Embed-only: {len(ready)}/{len(entries)} video(s) have TransNet metadata", flush=True)

    if args.phase in {"all", "embed"}:
        print("\n=== PHASE 2/2: CPU ffmpeg/preprocess producer + GPU PE-Core consumer ===")
        embedding_results, failures = run_embedding_phase(args, ready, failures)
    else:
        embedding_results = []

    summary = run_summary_path(args)
    atomic_json(summary, summary_payload(args, ready, embedding_results, failures))
    print(f"\nDone. Summary: {summary}")
    if failures:
        raise SystemExit(1)
