"""
Local (CPU-heavy) step: decode a video ONCE via ffmpeg (piped raw video, no
disk intermediate) and dump TransNetV2-ready low-res frames (27x48 RGB
uint8) as chunked .npz files for scp transfer to Colab.

Assumes constant frame rate (true for the competition video corpus): frame
index alone is enough downstream (scene selection, keyframe extraction), so
this does not scan per-frame PTS/timestamps -- that's what made the earlier
"merged Pass 1/2" decode two things in one ffmpeg call. Here there's only
ever one thing to decode, so a single `-vf scale` pass already gets the
speedup (measured 76.97s decode for 28498 frames, see REPORT.md exp #6).

Chunking exists only to make the transfer resumable/progress-visible; the
Colab side re-concatenates chunks into one array before running inference,
so there's no windowing/context loss at chunk boundaries.

Consecutive near-duplicate frames (mean-abs-diff below --dedup-threshold
against the last *kept* frame) are collapsed: only the first is stored, with
a repeat count. The Colab side reconstructs the original-length sequence via
np.repeat before running TransNetV2 -- this only shrinks the scp payload, it
does not change what TransNetV2 sees.

Usage:
  python local_extract_frames.py --video ../../../AIC-HCMC-2026/data/videos/L01_V002.mp4 --out-dir OUT [--start 0] [--end N] [--chunk-size 3000] [--dedup-threshold 1.0]

Sanity check before running on a full video (per CLAUDE.md "test small
before running big"): run with --end 200 first and confirm
num_frames == 200 in the printed meta.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
from pathlib import Path

import numpy as np

TRANSNET_W, TRANSNET_H = 48, 27
FRAME_BYTES = TRANSNET_W * TRANSNET_H * 3  # rgb24
STDERR_CAP_BYTES = 1 << 16  # cap in case ffmpeg is unexpectedly chatty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video", required=True,
        help="Video file path, or a zip_source.py subfile URL to decode straight out of a zip",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start", type=int, default=0, help="First frame index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Last frame index (exclusive); default = end of video")
    p.add_argument("--chunk-size", type=int, default=3000, help="Kept frames per chunk file")
    p.add_argument(
        "--dedup-threshold",
        type=float,
        default=1.0,
        help="Mean abs pixel diff (0-255) below which a frame is treated as a repeat of the last kept frame. 0 disables dedup.",
    )
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    return p.parse_args()


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def probe_fps(ffprobe_bin: str, video: str) -> float:
    out = subprocess.run(
        [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den) if den else float(num)


def drain_stderr(pipe, capped: list[bytes]) -> None:
    data = pipe.read(STDERR_CAP_BYTES)
    capped.append(data)
    # Discard anything beyond the cap so ffmpeg never blocks on a full pipe.
    while pipe.read(1 << 16):
        pass


def read_exact(pipe, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)  # EOF (possibly mid-frame)
        buf.extend(chunk)
    return bytes(buf)


def main() -> None:
    args = parse_args()
    if not args.video.startswith("subfile,,") and not Path(args.video).is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fps = probe_fps(args.ffprobe_bin, args.video)

    cmd = [
        args.ffmpeg_bin, "-v", "error", "-i", str(args.video),
        "-map", "0:v:0", "-an",
        "-vf", f"scale={TRANSNET_W}:{TRANSNET_H}:flags=area",
        "-pix_fmt", "rgb24",
    ]
    if args.start == 0 and args.end is not None:
        cmd += ["-frames:v", str(args.end)]  # fast path: let ffmpeg stop decoding early
    cmd += ["-f", "rawvideo", "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_capture: list[bytes] = []
    stderr_thread = threading.Thread(target=drain_stderr, args=(proc.stderr, stderr_capture), daemon=True)
    stderr_thread.start()

    chunk_frames: list[np.ndarray] = []
    chunk_repeats: list[int] = []
    last_kept: np.ndarray | None = None
    chunk_idx = 0
    n_raw = 0  # frames decoded (pre-dedup, pre --start skip)
    n_kept = 0  # frames written to disk (post-dedup)

    def flush_chunk() -> None:
        nonlocal chunk_idx, last_kept
        np.savez(
            args.out_dir / f"chunk_{chunk_idx:05d}.npz",
            frames=np.stack(chunk_frames),
            repeats=np.array(chunk_repeats, dtype=np.int32),
        )
        chunk_idx += 1
        chunk_frames.clear()
        chunk_repeats.clear()
        # Don't let a dedup run span chunks: chunk_repeats is now empty, so the
        # next frame must start a fresh entry rather than incrementing across
        # the boundary. Only costs a little compression at the seam.
        last_kept = None

    while args.end is None or n_raw < args.end:
        raw = read_exact(proc.stdout, FRAME_BYTES)
        if raw is None or len(raw) < FRAME_BYTES:
            break
        if n_raw < args.start:
            n_raw += 1
            continue
        frame_rgb = np.frombuffer(raw, dtype=np.uint8).reshape(TRANSNET_H, TRANSNET_W, 3)
        n_raw += 1

        if (
            last_kept is not None
            and args.dedup_threshold > 0
            and mean_abs_diff(frame_rgb, last_kept) <= args.dedup_threshold
        ):
            chunk_repeats[-1] += 1
            continue

        chunk_frames.append(frame_rgb.copy())  # copy: frombuffer view would go stale
        chunk_repeats.append(1)
        last_kept = frame_rgb
        n_kept += 1

        if len(chunk_frames) >= args.chunk_size:
            flush_chunk()

    if chunk_frames:
        flush_chunk()

    proc.stdout.close()
    returncode = proc.wait()
    stderr_thread.join(timeout=5)
    if returncode != 0:
        raise RuntimeError(f"ffmpeg exited {returncode}: {b''.join(stderr_capture).decode(errors='replace')}")

    meta = {
        "video": str(args.video),
        "fps": fps,
        "start_frame": args.start,
        "end_frame": args.start + (n_raw - args.start),
        "num_frames": n_raw - args.start,
        "num_kept_frames": n_kept,
        "num_chunks": chunk_idx,
        "chunk_size": args.chunk_size,
        "dedup_threshold": args.dedup_threshold,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {chunk_idx} chunk(s), {n_kept}/{meta['num_frames']} frames kept -> {args.out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
