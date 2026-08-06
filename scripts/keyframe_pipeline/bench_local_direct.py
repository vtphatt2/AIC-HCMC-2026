"""
Benchmark: decode + resize + TransNetV2 inference, all in-process, no network.
This is the "pure" baseline to compare against the local-extract -> scp ->
Colab-inference pipeline.

Usage:
  python bench_local_direct.py --video ../../../AIC-HCMC-2026/data/videos/L01_V002.mp4 --start 0 --end 3000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transnetv2_pytorch import TransNetV2

TRANSNET_SIZE = (48, 27)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None, help="Optional path to dump scene boundaries as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(args.video))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frames = []
    for _ in range(args.start, args.end):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(frame_bgr, TRANSNET_SIZE, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    decode_s = time.perf_counter() - t0

    frames_arr = np.stack(frames)
    t1 = time.perf_counter()
    model = TransNetV2()
    load_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    frames_t = torch.from_numpy(frames_arr).to(model.device)
    with torch.no_grad():
        single_frame_pred, _ = model.predict_frames(frames_t, quiet=True)
    infer_s = time.perf_counter() - t2

    scenes = model.predictions_to_scenes(single_frame_pred.cpu().numpy(), threshold=args.threshold)
    total_s = time.perf_counter() - t0

    print(f"frames={len(frames)} device={model.device}")
    print(f"decode+resize: {decode_s:.2f}s ({len(frames)/decode_s:.1f} fps)")
    print(f"model load:    {load_s:.2f}s")
    print(f"transnet infer:{infer_s:.2f}s ({len(frames)/infer_s:.1f} fps)")
    print(f"TOTAL:         {total_s:.2f}s")
    print(f"scenes: {len(scenes)}")

    if args.out:
        payload = {
            "device": str(model.device),
            "start_frame": args.start,
            "threshold": args.threshold,
            "scenes": [{"start_frame": int(s) + args.start, "end_frame": int(e) + args.start} for s, e in scenes],
        }
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
