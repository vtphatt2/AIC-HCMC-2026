"""
Local (CPU-heavy) step: decode a video ONCE via ffmpeg and materialize
full-resolution JPEGs for exactly the frame indices select_keyframes.py
chose (REPORT.md exp #5: 586 frames in 21.62s).

A single `select='eq(n,i1)+eq(n,i2)+...'` filter has a chain-length limit on
some local ffmpeg builds (~100-200 chained `eq(n,X)` terms -- see CLAUDE.md).
To stay under that while still decoding only once, the frame list is split
into <=80-frame groups, each its own `[0:v]split` branch with its own
`select`, each branch mapped to its own numbered output pattern -- so ffmpeg
does one decode pass total, just with N parallel select branches.

IMPORTANT (see CLAUDE.md): every `-map "[branch]"` must come AFTER
`-filter_complex`, one per output, each immediately followed by that
output's own options and destination pattern. Putting a `-map` before
`-filter_complex` previously leaked an extra full-res stream into an
output and wrote 33GB of garbage -- always sanity-test on a handful of
frames before trusting this on a full keyframe list.

Usage:
  python local_extract_keyframes.py --video ../../../AIC-HCMC-2026/data/videos/L01_V002.mp4 --keyframes KEYFRAMES/keyframes.json --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

GROUP_SIZE = 80  # stays comfortably under the ~100-200 chained eq(n,X) limit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video", required=True,
        help="Video file path, or a zip_source.py subfile URL to decode straight out of a zip",
    )
    p.add_argument("--keyframes", type=Path, required=True, help="keyframes.json from select_keyframes.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg -q:v (2=high quality, lower=better)")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    return p.parse_args()


def chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    args = parse_args()
    if not args.video.startswith("subfile,,") and not Path(args.video).is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    data = json.loads(args.keyframes.read_text())
    frame_numbers = sorted({kf["frame_number"] for kf in data["keyframes"]})
    if not frame_numbers:
        raise ValueError(f"No keyframes in {args.keyframes}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = chunked(frame_numbers, GROUP_SIZE)

    with tempfile.TemporaryDirectory(prefix="keyframe-extract-") as tmp_name:
        tmp_dir = Path(tmp_name)
        split_labels = [f"s{i}" for i in range(len(groups))]
        filter_parts = [f"[0:v]split={len(groups)}" + "".join(f"[{lbl}]" for lbl in split_labels)]
        for i, (lbl, group) in enumerate(zip(split_labels, groups)):
            expr = "+".join(f"eq(n\\,{n})" for n in group)
            filter_parts.append(f"[{lbl}]select='{expr}'[o{i}]")
        filter_complex = ";".join(filter_parts)

        cmd = [args.ffmpeg_bin, "-v", "error", "-i", str(args.video), "-an", "-filter_complex", filter_complex]
        for i in range(len(groups)):
            cmd += [
                "-map", f"[o{i}]",
                "-fps_mode", "passthrough",
                "-q:v", str(args.jpeg_quality),
                str(tmp_dir / f"g{i}_%06d.jpg"),
            ]

        subprocess.run(cmd, check=True)

        written = 0
        for i, group in enumerate(groups):
            decoded = sorted(tmp_dir.glob(f"g{i}_*.jpg"))
            if len(decoded) != len(group):
                raise RuntimeError(
                    f"group {i}: ffmpeg wrote {len(decoded)} frame(s), expected {len(group)} "
                    f"(frame numbers {group[0]}..{group[-1]})"
                )
            for frame_number, src in zip(group, decoded):
                dst = args.out_dir / f"frame_{frame_number:06d}.jpg"
                src.replace(dst)
                written += 1

    print(f"extracted {written}/{len(frame_numbers)} keyframe(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
