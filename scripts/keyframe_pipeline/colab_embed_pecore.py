"""
Colab (GPU-heavy) step: embed keyframe images with PE-Core-bigG-14-448.

Mirrors remote-server/app/services/text_encoder.py's torch backend (same
model_id, same normalize=True convention) but for images instead of text, so
vectors are directly comparable/compatible with the existing text encoder.

Usage (on the Colab VM):
  python3 colab_embed_pecore.py --images-dir KEYFRAMES --out-dir OUT [--batch-size 32]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"
EXPECTED_DIM = 1280


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ID, precision="fp32", device=args.device
    )
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"model load (incl. any weight download): {load_s:.1f}s device={args.device}")

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files in {args.images_dir}")

    t1 = time.perf_counter()
    n_done = 0
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(args.device)

        with torch.inference_mode():
            features = model.encode_image(batch, normalize=True)
        vectors = features.detach().float().cpu().numpy()

        for path, vector in zip(batch_paths, vectors):
            if vector.shape[0] != EXPECTED_DIM:
                raise ValueError(f"{path} produced dim {vector.shape[0]}, expected {EXPECTED_DIM}")
            np.save(args.out_dir / f"{path.stem}.npy", vector.astype("float32"))
        n_done += len(batch_paths)

    infer_s = time.perf_counter() - t1
    print(f"embedded {n_done} images in {infer_s:.1f}s ({n_done/infer_s:.1f} img/s)")


if __name__ == "__main__":
    main()
