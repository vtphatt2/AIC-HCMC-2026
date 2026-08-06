"""
Colab (GPU-heavy) step: concatenate chunked low-res frame .npz files back
into one array, re-expand deduped runs via np.repeat, and run TransNetV2
shot-boundary detection.

Usage (on the Colab VM):
  python3 colab_run_transnet.py --frames-dir FRAMES --out-dir OUT [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transnetv2_pytorch import TransNetV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads((args.frames_dir / "meta.json").read_text())

    chunk_paths = sorted(args.frames_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.npz files in {args.frames_dir}")
    kept_frames, repeats = [], []
    for p in chunk_paths:
        with np.load(p) as data:
            kept_frames.append(data["frames"])
            repeats.append(data["repeats"])
    kept_frames = np.concatenate(kept_frames, axis=0)
    repeats = np.concatenate(repeats, axis=0)
    if kept_frames.shape[0] != meta["num_kept_frames"]:
        raise ValueError(f"Loaded {kept_frames.shape[0]} kept frames, meta.json says {meta['num_kept_frames']}")

    frames = np.repeat(kept_frames, repeats, axis=0)
    if frames.shape[0] != meta["num_frames"]:
        raise ValueError(f"Reconstructed {frames.shape[0]} frames, meta.json says {meta['num_frames']}")

    model = TransNetV2()
    frames_t = torch.from_numpy(frames).to(model.device)

    with torch.no_grad():
        single_frame_pred, _ = model.predict_frames(frames_t)

    single_np = single_frame_pred.cpu().numpy()
    local_scenes = model.predictions_to_scenes(single_np, threshold=args.threshold)

    offset = meta["start_frame"]
    scenes = [
        {"start_frame": int(s) + offset, "end_frame": int(e) + offset}
        for s, e in local_scenes
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "predictions.npy", single_np)
    result = {
        "video": meta["video"],
        "fps": meta["fps"],
        "start_frame": meta["start_frame"],
        "end_frame": meta["end_frame"],
        "threshold": args.threshold,
        "device": str(model.device),
        "num_scenes": len(scenes),
        "scenes": scenes,
    }
    (args.out_dir / "scenes.json").write_text(json.dumps(result, indent=2))
    print(f"device={model.device} frames={frames.shape[0]} scenes={len(scenes)}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
